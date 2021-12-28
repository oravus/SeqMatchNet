import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import torch
import time

class seqMatchNet(nn.Module):

    def __init__(self):
        super(seqMatchNet, self).__init__()

    def cdist_quick(self,r,c):
        return torch.sqrt(2 - 2*torch.matmul(r,c.transpose(1,2)))

    def aggregateSeqScore(self,data):
        r, c, method = data
        dMat = self.cdist_quick(r,c)
        seqL = dMat.shape[1]
        dis = torch.diagonal(dMat,0,1,2)
        if method == 'centerOnly':
            dis = dis[:,seqL//2]
        else: # default to 'meanOfPairs'
            dis = dis.mean(-1)
        return dis

    def forward(self,data):
        return self.aggregateSeqScore(data)

    def computeDisMat_torch(r,c):
        # assumes descriptors to be l2-normalized
        return torch.stack([torch.sqrt(2 - 2*torch.matmul(r,c[i].unsqueeze(1))).squeeze() for i in range(c.shape[0])]).transpose(0,1)

def modInd(idx,l,n):
    return max(l,min(idx,n-l-1))

def computeRange(l,n):
    li, le = l//2, l-l//2
    return torch.stack([torch.arange(modInd(r,li,n)-li,modInd(r,li,n)+le,dtype=int) for r in range(n)])

def aggregateMatchScores_pt_fromMat_oneShot(dMat,l,device):
    convWeight = torch.eye(l,device=device).unsqueeze(0).unsqueeze(0)
    dMat_seq = -1*torch.ones(dMat.shape,device=device)
    li, le = l//2, l-l//2

    dMat_seq[li:-le+1,li:-le+1] = torch.nn.functional.conv2d(dMat.unsqueeze(0).unsqueeze(0),convWeight).squeeze()

    # fill left and right columns
    dMat_seq[:,:li] = dMat_seq[:,li,None]
    dMat_seq[:,-le+1:] = dMat_seq[:,-le,None]

    # fill top and bottom rows
    dMat_seq[:li,:] = dMat_seq[None,li,:]
    dMat_seq[-le+1:,:] = dMat_seq[None,-le,:]

    return dMat_seq

def aggregateMatchScores_pt_fromMat(dMat,l,device,refCandidates=None):
    li, le = l//2, l-l//2
    n = dMat.shape[0]
    convWeight = torch.eye(l,device=device).unsqueeze(0).unsqueeze(0)

#     dMat = dMat.to('cpu')
    if refCandidates is None:
        shape = dMat.shape
    else:
        shape = refCandidates.transpose().shape
        preCompInds = computeRange(l,n)

    dMat_seq = -1*torch.ones(shape,device=device)

    durs = []
    for j in tqdm(range(li,dMat.shape[1]-li), total=dMat.shape[1]-l, leave=True):
        t1 = time.time()
        if refCandidates is not None:
            rCands = preCompInds[refCandidates[j]].flatten()
            dMat_cols = dMat[rCands,j-li:j+le].to(device)
            dMat_seq[:,j] = torch.nn.functional.conv2d(dMat_cols.unsqueeze(0).unsqueeze(0),convWeight,stride=l).squeeze()
        else:
            dMat_cols = dMat[:,j-li:j+le].to(device)
            dMat_seq[li:-le+1,j] = torch.nn.functional.conv2d(dMat_cols.unsqueeze(0).unsqueeze(0),convWeight).squeeze()
        durs.append(time.time()-t1)

    if refCandidates is None:
        # fill left and right columns
        dMat_seq[:,:li] = dMat_seq[:,li,None]
        dMat_seq[:,-le+1:] = dMat_seq[:,-le,None]

        # fill top and bottom rows
        dMat_seq[:li,:] = dMat_seq[None,li,:]
        dMat_seq[-le+1:,:] = dMat_seq[None,-le,:]

#     assert(np.sum(dMat_seq==-1)==0)
    print("Average Time Per Query", np.mean(durs))
    return dMat_seq

def aggregateMatchScores_pt_fromDesc(dbDesc,qDesc,l,device,refCandidates=None):
    numDb, numQ = dbDesc.shape[0], qDesc.shape[0]
    convWeight = torch.eye(l,device=device).unsqueeze(0).unsqueeze(0)

    if refCandidates is None:
        shape = [numDb,numQ]
    else:
        shape = refCandidates.transpose().shape

    dMat_seq = -1*torch.ones(shape,device=device)

    durs = []
    for j in tqdm(range(numQ), total=numQ, leave=True):
        t1 = time.time()
        if refCandidates is not None:
            rCands = refCandidates[j]
        else:
            rCands = torch.arange(numDb)

        dMat = torch.cdist(dbDesc[rCands],qDesc[j].unsqueeze(0))
        dMat_seq[:,j] = torch.nn.functional.conv2d(dMat.unsqueeze(1),convWeight).squeeze()
        durs.append(time.time()-t1)

#     assert(torch.sum(dMat_seq==-1)==0)
    print("Average Time Per Query", np.mean(durs), np.std(durs))
    return dMat_seq

def aggregateMatchScores(dMat,l,device='cuda',refCandidates=None,dbDesc=None,qDesc=None,dMatProcOneShot=False):
    dMat_seq, matchInds, matchDists = None, None, None
    if dMat is None:
        dMat_seq = aggregateMatchScores_pt_fromDesc(dbDesc,qDesc,l,device,refCandidates).detach().cpu().numpy()
    else:
        if dMatProcOneShot:
            dMat_seq = aggregateMatchScores_pt_fromMat_oneShot(dMat,l,device).detach().cpu().numpy()
        else:
            dMat_seq = aggregateMatchScores_pt_fromMat(dMat,l,device,refCandidates).detach().cpu().numpy()
    return dMat_seq, matchInds, matchDists
