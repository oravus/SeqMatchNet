import numpy as np
import torch
import faiss
from os.path import join
import shutil

import seqMatchNet

N_VALUES = [1,5,10,20,100]

def batch2Seq(input,l):
    inSh = input.shape
    input = input.view(inSh[0]//l,l,inSh[1])
    return input

def seq2Batch(input):
    inSh = input.shape
    input = input.view(inSh[0]*inSh[1],inSh[2],inSh[3],inSh[4])
    return input

def save_checkpoint(savePath, state, is_best, filename='checkpoint.pth.tar'):
    model_out_path = join(savePath, filename)
    torch.save(state, model_out_path)
    if is_best:
        shutil.copyfile(model_out_path, join(savePath, 'model_best.pth.tar'))
        
def getRecallAtN(n_values, predictions, gt):
    correct_at_n = np.zeros(len(n_values))
    numQWithoutGt = 0
    #TODO can we do this on the matrix in one go?
    for qIx, pred in enumerate(predictions):
        if len(gt[qIx]) == 0:
            numQWithoutGt += 1
            continue
        for i,n in enumerate(n_values):
            # if in top N then also in top NN, where NN > N
            if np.any(np.in1d(pred[:n], gt[qIx])):
                correct_at_n[i:] += 1
                break
#     print("Num Q without GT: ", numQWithoutGt, " of ", len(gt))
    return correct_at_n / (len(gt)-numQWithoutGt)

def computeMatches(opt,n_values,device,dbFeat=None,qFeat=None,dbFeat_np=None,qFeat_np=None,dMat=None):

    if opt.matcher is not None:
        if dMat is None:
            if opt.predictionsFile is not None:
                predPrior = np.load(opt.predictionsFile)['preds']
                predPriorTopK = predPrior[:,:20]
            else:
                predPriorTopK = None
            outSeqL = opt.seqL
            dMat = 1.0/outSeqL * seqMatchNet.aggregateMatchScores(None,outSeqL,device, dbDesc=dbFeat, qDesc=qFeat,refCandidates=predPriorTopK)[0]
        print(dMat.shape)
        predictions = np.argsort(dMat,axis=0)[:max(n_values),:].transpose()
        bestDists = dMat[predictions[:,0],np.arange(dMat.shape[1])]
        if opt.predictionsFile is not None:
            predictions = np.array([predPriorTopK[qIdx][predictions[qIdx]] for qIdx in range(predictions.shape[0])])
            print("Preds:",predictions.shape)

    # single image descriptors
    else:
        assert(opt.seqL==1)
        print('====> Building faiss index')
        faiss_index = faiss.IndexFlatL2(dbFeat_np.shape[-1])
        faiss_index.add(np.squeeze(dbFeat_np))

        distances, predictions = faiss_index.search(np.squeeze(qFeat_np), max(n_values))
        bestDists = distances[:,0]
    return predictions, bestDists

def evaluate(n_values,predictions,gtDistsMat=None):
    print('====> Calculating recall @ N')
    # compute recall for different loc radii
    rAtL = []
    for locRad in [1,5,10,20,40,100,200]:
        gtAtL = gtDistsMat <= locRad
        gtAtL = [np.argwhere(gtAtL[:,qIx]).flatten() for qIx in range(gtDistsMat.shape[1])]
        rAtL.append(getRecallAtN(n_values, predictions, gtAtL))
    return rAtL