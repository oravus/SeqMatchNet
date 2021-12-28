import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset
from tqdm import tqdm
import numpy as np
from os import remove
import h5py
from math import ceil
import wandb
from termcolor import colored

from utils import batch2Seq, seq2Batch, getRecallAtN, computeMatches, N_VALUES
import seqMatchNet

def train(opt, model, encoder_dim, device, dataset, criterion, optimizer, train_set, whole_train_set, whole_training_data_loader, epoch, writer):
    epoch_loss = 0
    startIter = 1 # keep track of batch iter across subsets for logging

    if opt.cacheRefreshRate > 0:
        subsetN = ceil(len(train_set) / opt.cacheRefreshRate)
        #TODO randomise the arange before splitting?
        subsetIdx = np.array_split(np.arange(len(train_set)), subsetN)
    else:
        subsetN = 1
        subsetIdx = [np.arange(len(train_set))]

    nBatches = (len(train_set) + opt.batchSize - 1) // opt.batchSize

    for subIter in range(subsetN):
        print('====> Building Cache')
        model.eval()
        with h5py.File(train_set.cache, mode='w') as h5: 
            pool_size = encoder_dim
            validInds = whole_train_set.validInds
            h5feat = h5.create_dataset("features", [len(validInds), pool_size], dtype=np.float32)
            h5DisMat = h5.create_dataset("disMat",[whole_train_set.numDb_valid, whole_train_set.numQ_valid],dtype=np.float32)
            with torch.no_grad():
                dbFeat_single = torch.zeros(len(whole_train_set), pool_size, device=device)
                # expected input B,T,C,H,W (T is 1)
                for iteration, (input, indices) in tqdm(enumerate(whole_training_data_loader, 1),total=len(whole_training_data_loader)-1, leave=True):
                    # convert to B*T,C,H,W
                    image_encoding = seq2Batch(input).float().to(device)

                    # input B*T,C,1,1; outputs B,T,C (T=1); squeeze to B,C
                    global_single_descs = model.pool(image_encoding).squeeze()
                    dbFeat_single[indices] = global_single_descs
                del input, image_encoding, global_single_descs

                outSeqL = opt.seqL
                # fill sequences centered at single images
                dbFeat = torch.zeros(len(validInds), outSeqL, pool_size, device=device)
                for ind in range(len(validInds)):
                    dbFeat[ind] = dbFeat_single[whole_train_set.getSeqIndsFromValidInds(validInds[ind])]
                    if opt.matcher is None: # assumes seqL is 1 in this case
                        h5feat[ind] = dbFeat[ind].squeeze()
                del dbFeat_single

                if opt.matcher is not None:
                    offset = whole_train_set.numDb_valid
                    # compute distance matrix
                    print('====> Caching distance matrix')
                    if opt.matcher == 'seqMatchNet':
                        if 'nordland' in opt.dataset.lower() or 'tmr' in opt.dataset.lower():
                            dMat_cont = seqMatchNet.seqMatchNet.computeDisMat_torch(dbFeat[:offset,outSeqL//2], dbFeat[offset:,outSeqL//2])
                            if opt.neg_trip_method == 'centerOnly':
                                h5DisMat[...] = dMat_cont.detach().cpu().numpy()
                            elif opt.neg_trip_method == 'meanOfPairs':
                                h5DisMat[...] = seqMatchNet.aggregateMatchScores(dMat_cont,outSeqL,device,dMatProcOneShot=False)[0] * (1.0/outSeqL)
                        else:
                            if opt.neg_trip_method == 'centerOnly':
                                h5DisMat[...] = seqMatchNet.aggregateMatchScores(None,1,device, dbDesc=dbFeat[:offset,outSeqL//2:outSeqL//2+1], qDesc=dbFeat[offset:,outSeqL//2:outSeqL//2+1])[0]
                            elif opt.neg_trip_method == 'meanOfPairs':
                                h5DisMat[...] = seqMatchNet.aggregateMatchScores(None,outSeqL,device, dbDesc=dbFeat[:offset], qDesc=dbFeat[offset:])[0] * (1.0/outSeqL)
                    else:
                        raise("TODO")

                    del dbFeat
                    dMat = h5DisMat[()]
            dbFeat_np, qFeat_np = h5feat[:offset].copy(), h5feat[offset:].copy()

        sub_train_set = Subset(dataset=train_set, indices=subsetIdx[subIter])

        training_data_loader = DataLoader(dataset=sub_train_set, num_workers=opt.threads, 
                    batch_size=opt.batchSize, shuffle=True, 
                    collate_fn=dataset.collate_fn, pin_memory=False)

        if not opt.nocuda:
            print('Allocated:', torch.cuda.memory_allocated())
            print('Cached:', torch.cuda.memory_reserved())

        print('====> Training Queries')
        model.train()
        for iteration, (query, positives, negatives, 
                negCounts, indices) in tqdm(enumerate(training_data_loader, startIter),total=len(training_data_loader),leave=True):
            loss = 0
            if query is None:
                continue # in case we get an empty batch

            B, C = len(query), query[0].shape[1]
            nNeg = torch.sum(negCounts)
            image_encoding = seq2Batch(torch.cat([query, positives, negatives])).float()

            image_encoding = image_encoding.to(device)
            global_single_descs = model.pool(image_encoding)
            global_single_descs = batch2Seq(global_single_descs.squeeze(1),opt.seqL)

            del image_encoding
            g_desc_Q, g_desc_P, g_desc_N = torch.split(global_single_descs, [B, B, nNeg])
            del global_single_descs
            optimizer.zero_grad()
            
            # calculate loss for each Query, Positive, Negative triplet
            # due to potential difference in number of negatives have to
            # do it per query, per negative
            trips_a, trips_p, trips_n = [], [], []
            for i, negCount in enumerate(negCounts):
                for n in range(negCount):
                    negIx = (torch.sum(negCounts[:i]) + n).item()
                    if opt.matcher is None:
                        loss += criterion(g_desc_Q[i:i+1].squeeze(1), g_desc_P[i:i+1].squeeze(1), g_desc_N[negIx:negIx+1].squeeze(1))
                    else:
                        trips_a.append(g_desc_Q[i:i+1])
                        trips_p.append(g_desc_P[i:i+1])
                        trips_n.append(g_desc_N[negIx:negIx+1])

            del g_desc_Q, g_desc_P, g_desc_N
            if opt.matcher is not None:
                dis_ap = model.matcher([torch.cat(trips_a), torch.cat(trips_p), opt.loss_trip_method])
                dis_an = model.matcher([torch.cat(trips_a), torch.cat(trips_n), opt.loss_trip_method])
                loss = torch.max(dis_ap - dis_an + opt.margin**0.5,torch.zeros(dis_ap.shape,device=device)).mean()
                del trips_a, trips_p, trips_n
            else:
                loss /= nNeg.float().to(device) # normalise by actual number of negatives
            loss.backward()
            optimizer.step()

            batch_loss = loss.item()
            epoch_loss += batch_loss

            if iteration % 50 == 0 or nBatches <= 10:
                print("==> Epoch[{}]({}/{}): Loss: {:.4f}".format(colored(epoch,'red'), iteration,
                    nBatches, batch_loss), flush=True)
                writer.add_scalar('Train/Loss', batch_loss, 
                        ((epoch-1) * nBatches) + iteration)
                writer.add_scalar('Train/nNeg', nNeg, 
                        ((epoch-1) * nBatches) + iteration)
                wandb.log({"loss":batch_loss, "nNeg":nNeg, "epoch":epoch})
                if not opt.nocuda:
                    print('Allocated:', torch.cuda.memory_allocated())
                    print('Cached:', torch.cuda.memory_reserved())

            del query, positives, negatives

        startIter += len(training_data_loader)
        del training_data_loader, loss
        optimizer.zero_grad()
        torch.cuda.empty_cache()
        remove(train_set.cache) # delete HDF5 cache

    avg_loss = epoch_loss / nBatches

    print("===> Epoch {} Complete: Avg. Loss: {:.4f}".format(colored(epoch,'red'), avg_loss),
            flush=True)
    writer.add_scalar('Train/AvgLoss', avg_loss, epoch)
    predictions, bestDists = computeMatches(opt,N_VALUES,device,dbFeat_np=dbFeat_np,qFeat_np=qFeat_np,dMat=dMat)
    gt,gtDists = whole_train_set.get_positives(retDists=True)
    recall_at_n = getRecallAtN(N_VALUES, predictions, gt)
    wandb.log({"loss_e":avg_loss},commit=False)
    for i,n in enumerate(N_VALUES):
        writer.add_scalar('Train/Recall@' + str(n), recall_at_n[i], epoch)
        wandb.log({'Train/Recall@' + str(n): recall_at_n[i]},commit=False)