import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.manifold import TSNE
from scipy.spatial.distance import cdist
import numpy as np
import time
import wandb

from utils import seq2Batch, getRecallAtN, computeMatches, evaluate, N_VALUES

def test(opt, model, encoder_dim, device, eval_set, writer, epoch=0, extract_noEval=False):
    # TODO what if features dont fit in memory? 
    test_data_loader = DataLoader(dataset=eval_set, 
                num_workers=opt.threads, batch_size=opt.cacheBatchSize, shuffle=False, 
                pin_memory=False)

    model.eval()
    with torch.no_grad():
        print('====> Extracting Features')
        pool_size = encoder_dim
        validInds = eval_set.validInds
        dbFeat_single = torch.zeros((len(eval_set), pool_size),device=device)
        durs_batch = []
        for iteration, (input, indices) in tqdm(enumerate(test_data_loader, 1),total=len(test_data_loader)-1, leave=False):
            t1 = time.time()
            image_encoding = seq2Batch(input).float().to(device)
            global_single_descs = model.pool(image_encoding).squeeze()
            dbFeat_single[indices] = global_single_descs

            if iteration % 50 == 0 or len(test_data_loader) <= 10:
                print("==> Batch ({}/{})".format(iteration, 
                    len(test_data_loader)), flush=True)
            durs_batch.append(time.time()-t1)
        del input, image_encoding, global_single_descs

    del test_data_loader
    print("Average batch time:", np.mean(durs_batch), np.std(durs_batch))

    outSeqL = opt.seqL
    # use the original single descriptors for fast seqmatch over dMat (MSLS-like non-continuous dataset not supported)
    if (not opt.pooling and opt.matcher is None) and ('nordland' in opt.dataset.lower() or 'tmr' in opt.dataset.lower() or 'ox' in opt.dataset.lower()):
        dbFeat = dbFeat_single
        numDb = eval_set.numDb_full
    # fill sequences centered at single images
    else:
        dbFeat = torch.zeros(len(validInds), outSeqL, pool_size, device=device)
        numDb = eval_set.dbStruct.numDb
        for ind in range(len(validInds)):
            dbFeat[ind] = dbFeat_single[eval_set.getSeqIndsFromValidInds(validInds[ind])]
    del dbFeat_single

    # extracted for both db and query, now split in own sets
    qFeat = dbFeat[numDb:]
    dbFeat = dbFeat[:numDb]
    print(dbFeat.shape, qFeat.shape)

    qFeat_np = qFeat.detach().cpu().numpy().astype('float32')
    dbFeat_np = dbFeat.detach().cpu().numpy().astype('float32')

    db_emb, q_emb = None, None
    if opt.numSamples2Project != -1 and writer is not None:
        db_emb = TSNE(n_components=2).fit_transform(dbFeat_np[:opt.numSamples2Project])
        q_emb = TSNE(n_components=2).fit_transform(qFeat_np[:opt.numSamples2Project])

    if extract_noEval:
        return np.vstack([dbFeat_np,qFeat_np]), db_emb, q_emb, None, None

    predictions, bestDists = computeMatches(opt,N_VALUES,device,dbFeat,qFeat,dbFeat_np,qFeat_np)

    # for each query get those within threshold distance
    gt,gtDists = eval_set.get_positives(retDists=True)
    gtDistsMat = cdist(eval_set.dbStruct.utmDb,eval_set.dbStruct.utmQ)

    recall_at_n = getRecallAtN(N_VALUES, predictions, gt)
    rAtL = evaluate(N_VALUES,predictions,gtDistsMat)

    recalls = {} #make dict for output
    for i,n in enumerate(N_VALUES):
        recalls[n] = recall_at_n[i]
        print("====> Recall@{}: {:.4f}".format(n, recall_at_n[i]))
        if writer is not None:
            writer.add_scalar('Val/Recall@' + str(n), recall_at_n[i], epoch)
        wandb.log({'Val/Recall@' + str(n): recall_at_n[i]},commit=False)

    return recalls, db_emb, q_emb, rAtL, predictions
