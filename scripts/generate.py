import os
import logging
import time
from datetime import datetime
from typing import Optional, Literal, List
import argparse

from sklearn.ensemble import RandomForestClassifier
from joblib import dump, load

import torch
from torch.utils.data import Dataset
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import pandas as pd

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit.DataStructs import ConvertToNumpyArray
from rdkit.ML.Cluster import Butina

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

logger = logging.getLogger("generate")
logger.setLevel(logging.DEBUG)
# create file handler which logs even debug messages
fh = logging.FileHandler('tmp.log')
fh.setLevel(logging.DEBUG)
logger.addHandler(fh)

TOKENS = (
    'X', '<', '>', '#', '%', ')', '(', '+', '-', '/', '.', '1', '0', '3', '2',
    '5', '4', '7', '6', '9', '8', '=', 'A', '@', 'C', 'B', 'F', 'I', 'H', 'O',
    'N', 'P', 'S', '[', ']', '\\', 'c', 'e', 'i', 'l', 'o', 'n', 'p', 's', 'r',
    '\n'
)


# HELPER UTIL FUNCTIONS


def to_item(lst):
    if isinstance(lst, list):
        return lst[0]
    if isinstance(lst, np.ndarray):
        return lst[0]
    if isinstance(lst, pd.Series):
        return lst.values[0]
    return lst


def calc_rdkit(mol, min_path=1, max_path=7, n_bits=2048, n_bits_per_hash=2):
    mol = to_item(mol)

    # if bad mol, return array of nan (so it can still fit in the array)
    if mol is None:
        return np.full(n_bits, np.nan)

    _fp = Chem.RDKFingerprint(mol, minPath=min_path, maxPath=max_path, fpSize=n_bits, nBitsPerHash=n_bits_per_hash)

    fp = np.zeros(n_bits, dtype=np.int8)
    ConvertToNumpyArray(_fp, fp)

    return fp


def calc_morgan(mol, radius=4, n_bits=256, count=False, use_chirality=True):
    mol = to_item(mol)

    # if bad mol, return array of nan (so it can still fit in the array)
    if mol is None:
        return np.full(n_bits, np.nan)

    if count:
        _fp = AllChem.GetHashedMorganFingerprint(mol, radius=radius, nBits=n_bits, useChirality=use_chirality)
    else:
        _fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits, useChirality=use_chirality)

    return _fp


def get_fps(smis, y, func: Literal["morgan", "rdkit"]):
    mols = [Chem.MolFromSmiles(smile) for smile in smis]
    if y is not None:
        mols = [(m,y) for (m,y) in zip(mols, y) if m is not None]# can ghost remove but to bad
        y = np.array([_[1] for _ in mols])
        mols = [_[0] for _ in mols]
    else:
        mols = [m for m in mols if m is not None]
    if func == "morgan":
        return [calc_morgan(c) for c in mols], y
    if func == "rdkit":
        return np.array([calc_rdkit(c) for c in mols]), y
    raise ValueError(f"cannot find func {func}")


def flatten(smis, y):
    mols = []
    for i, smile in enumerate(smis):
        mols.append(Chem.MolFromSmiles(smile))
        if ((i+1) % 100000 == 0):
            logger.info(f"finished {i}")
    if y is not None:
        mols = [(m, y) for (m, y) in zip(mols, y) if m is not None]# can ghost remove but to bad
        y = np.array([_[1] for _ in mols])
        mols = [_[0] for _ in mols]
    else:
        mols = [m for m in mols if m is not None]
    return [Chem.MolToSmiles(mol, isomericSmiles=False) for mol in mols], y


def load_smiles(file_path, smi_col, label_col=None, delimiter=",", header=True):
    df = pd.read_csv(file_path, delimiter=delimiter, header=0 if header else None)

    smiles = df[smi_col]

    if label_col is not None:
        labels = df[label_col]
        return smiles, labels

    return smiles


def bulk_fp_tanimoto_similarity(smiles_fps1, smiles_fps2, pooling: Optional[Literal['max', 'mean']] = None):
    if len(smiles_fps2) == 0:
        return np.zeros(len(smiles_fps1))
    dists = []
    for i, s1 in enumerate(smiles_fps1):
        dists.append(DataStructs.BulkTanimotoSimilarity(s1, smiles_fps2))
    dists = np.array(dists)
    if pooling is None:
        return dists
    if pooling == "max":
        return np.max(dists, axis=1)
    elif pooling == "mean":
        return np.mean(dists, axis=1)
    else:
        raise ValueError(f"pooling must be in ['max', 'mean'] got {pooling}")


def pdist_tanimoto(smiles):
    fps = get_fps(smiles, func="morgan")
    dists = []
    for i in range(1, len(fps)):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i],fps[:i])
        dists.extend([1-x for x in sims])
    return dists


def cluster_butina(chemicals, dist_thresh=0.2, **kwargs):
    return Butina.ClusterData(pdist_tanimoto(chemicals), len(chemicals), distThresh=dist_thresh, isDistData=True)


class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

        if mask is not None:
            attn = attn.masked_fill(mask == 1, -1e9)

        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)

        return output, attn


class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5,
                                                   attn_dropout=dropout)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, q, k, v, mask=None):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)  # For head axis broadcasting.

        q, attn = self.attention(q, k, v, mask=mask)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads
        # together: b x lq x (n*dv)
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        q = self.dropout(self.fc(q))
        q += residual

        q = self.layer_norm(q)

        return q, attn


class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid)  # position-wise
        self.w_2 = nn.Linear(d_hid, d_in)  # position-wise
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x

        x = self.w_2(F.relu(self.w_1(x)))
        x = self.dropout(x)
        x += residual

        x = self.layer_norm(x)

        return x


class EncoderLayer(nn.Module):
    ''' Compose with two layers '''

    def __init__(self, d_model, d_inner, n_head, d_k, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_k,
                                           dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner,
                                               dropout=dropout)

    def forward(self, enc_input, slf_attn_mask=None):
        enc_output, enc_slf_attn = self.slf_attn(
            enc_input, enc_input, enc_input, mask=slf_attn_mask)

        enc_output = self.pos_ffn(enc_output)
        return enc_output, enc_slf_attn


def get_subsequent_mask(seq):
    ''' For masking out the subsequent info. '''
    sz_b, len_s = seq.size()
    subsequent_mask = torch.triu(torch.ones((len_s, len_s),
                                            device=seq.device),
                                 diagonal=1).unsqueeze(0).byte()
    return subsequent_mask


class PositionalEncoding(nn.Module):

    def __init__(self, d_hid, n_position=200):
        super(PositionalEncoding, self).__init__()

        # Not a parameter
        self.register_buffer('pos_table',
                             self._get_sinusoid_encoding_table(n_position,
                                                               d_hid))

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        ''' Sinusoid position encoding table '''

        # TODO: make it with torch instead of numpy

        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) \
                    for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) \
                                   for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, x):
        return x + self.pos_table[:, :x.size(1)].clone().detach()


class Encoder(nn.Module):
    ''' A encoder model with self attention mechanism. '''

    def __init__(
            self, n_src_vocab, d_word_vec, n_layers, n_head, d_k,
            d_inner, pad_idx=None, dropout=0.1, n_position=200):

        super().__init__()

        self.src_word_emb = nn.Embedding(n_src_vocab, d_word_vec,
                                         padding_idx=pad_idx)
        self.position_enc = PositionalEncoding(d_word_vec,
                                               n_position=n_position)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_word_vec, d_inner, n_head,
                         d_k, dropout=dropout)
            for _ in range(n_layers)])
        self.layer_norm = nn.LayerNorm(d_word_vec, eps=1e-6)

    def forward(self, src_seq, src_mask, return_attns=False):

        enc_slf_attn_list = []

        # -- Forward

        enc_output = \
            self.dropout(self.position_enc(self.src_word_emb(src_seq)))
        enc_output = self.layer_norm(enc_output)

        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(enc_output,
                                                 slf_attn_mask=src_mask)
            enc_slf_attn_list += [enc_slf_attn] if return_attns else []

        if return_attns:
            return enc_output, enc_slf_attn_list
        return enc_output


class Decoder(nn.Module):

    def __init__(self, n_src_vocab, d_word_vec, dropout=0.1):
        super().__init__()

        self.decoder = nn.Linear(d_word_vec, n_src_vocab)

    def forward(self, enc_output):
        return self.decoder(enc_output)


class Transformer(nn.Module):

    def __init__(self, n_src_vocab, d_word_vec, n_layers, n_head, d_k,
                 d_inner, pad_idx=None, dropout=0.1, n_position=200):

        super().__init__()

        self.start_token = '<'
        self.end_token = '>'
        self.pad_token = 'X'
        self.max_len = 120

        self.encoder = Encoder(n_src_vocab, d_word_vec, n_layers, n_head, d_k,
                               d_inner, pad_idx, dropout, n_position)

        self.decoder = Decoder(n_src_vocab, d_word_vec)

    def forward(self, seq, return_attns=False, return_enc=False):

        if return_attns:
            enc_out, attns = self.encoder(seq,
                                          get_subsequent_mask(seq),
                                          return_attns)
            return self.decoder(enc_out), attns

        elif return_enc:
            enc_out = self.encoder(seq, get_subsequent_mask(seq),
                                   return_attns)
            return self.decoder(enc_out), enc_out

        else:
            return self.decoder(self.encoder(seq, get_subsequent_mask(seq),
                                             return_attns))

    def count_params(self):
        return sum([param.nelement() for param in self.parameters()])

    def fit(self, loader, optimizer, scheduler, n_epochs, n_steps=-1, use_cuda=True, quiet=True, device="cuda:0"):
        all_losses = []

        device = device
        data_type = torch.int64

        self.to(device)
        self.train()

        for epoch in range(n_epochs):
            for i, seq in enumerate(loader):
                if (n_steps != -1) & (i >= n_steps - 1):
                    break

                seq = seq.to(device, dtype=data_type)

                optimizer.zero_grad()

                pad_mask = torch.where(seq == TOKENS.index(self.pad_token), torch.zeros_like(seq), torch.ones_like(seq))
                pad_mask = torch.flatten(pad_mask[:, :-1]).float()
                criterion = nn.CrossEntropyLoss(reduction='none')
                unred_loss = criterion(torch.flatten(self(seq)[:, :-1], end_dim=-2), torch.flatten(seq[:, 1:]))

                loss = torch.matmul(unred_loss, pad_mask) / torch.sum(pad_mask)
                loss.backward()
                optimizer.step()
                all_losses.append(loss.item())

            if scheduler:
                scheduler.step()
        return all_losses

    def generate(self, batch_size, use_cuda=True, device="cuda:0"):
        device = torch.device("cuda" if use_cuda else "cpu")

        start_idx = TOKENS.index(self.start_token)
        end_idx = TOKENS.index(self.end_token)
        pad_idx = TOKENS.index(self.pad_token)
        src = torch.ones(len(TOKENS))
        idx_tens = torch.tensor([start_idx, pad_idx])

        mask = torch.stack([torch.zeros(len(TOKENS)).scatter_(0, idx_tens, src)
                            for _ in range(batch_size)]).to(device)

        self.to(device)
        self.eval()

        start_batch = [start_idx for _ in range(batch_size)]
        seq = torch.tensor(start_batch).unsqueeze(1).to(device)

        with torch.no_grad():
            for p in range(self.max_len):
                logits = self(seq)[:, -1].masked_fill(mask == 1, -1e9)
                top_i = torch.distributions.categorical.Categorical(logits=logits).sample()
                top_i = top_i.masked_fill((seq[:, -1] == end_idx) | (seq[:, -1] == pad_idx), pad_idx)
                seq = torch.cat([seq, top_i.unsqueeze(1)], dim=-1)
            close_seq = torch.tensor([end_idx for _ in range(batch_size)]).to(device)
            close_seq = close_seq.masked_fill((seq[:, -1] == end_idx) | (seq[:, -1] == pad_idx), pad_idx)
            seq = torch.cat([seq, close_seq.unsqueeze(1)], dim=-1)

        return np.array([''.join([TOKENS[seq[i, j]] for j in range(self.max_len + 2) if seq[i, j] != pad_idx][1:-1])
                         for i in range(batch_size)])



class _BaseChip:
    def __init__(self, use_cuda, tokens=TOKENS, start_token='<',
                 end_token='>', pad_token='X', max_len=120, device="cuda:0"):
        super().__init__()

        self.start_token = start_token
        self.end_token = end_token
        self.pad_token = pad_token

        self.all_characters = tokens
        self.char2idx = dict((token, i) for i, token in enumerate(tokens))
        self.n_characters = len(tokens)

        self.pad_idx = self.all_characters.index(self.pad_token)

        self.max_len = max_len

        self.use_cuda = use_cuda
        if self.use_cuda is None:
            self.use_cuda = torch.cuda.is_available()

        self._device = device

    def _vectorify(self, smiles):
        vectors = []
        for i in range(len(smiles)):
            if len(smiles[i]) <= self.max_len:
                pad_seq = ''.join([self.pad_token for _ in range(self.max_len - len(smiles[i]))])
                vectors.append(self.start_token + smiles[i] + self.end_token + pad_seq)
        return vectors

    def _pad_and_cap_str(self, string, max_len):
        pad_seq = ''.join([self.pad_token for _ in range(max_len - len(string))])

        new_string = self.start_token + string + self.end_token + pad_seq

        return new_string

    def _char_tensor(self, string):
        """
        Converts SMILES into tensor of indices wrapped into
        torch.autograd.Variable.
        Args:
            string (str): input SMILES string
        Returns:
            tokenized_string (torch.autograd.Variable(torch.tensor))
        """
        tensor = torch.zeros(len(string)).to(self._device).long()
        for c in range(len(string)):
            tensor[c] = self.all_characters.index(string[c])
        return tensor.to(self._device)


class ChipGenerateDataset(Dataset, _BaseChip):

    def __init__(self, smiles, tokens=TOKENS, start_token='<',
                 end_token='>', pad_token='X', max_len=120, use_cuda=True, device="cuda:0"):
        """
        Constructor for the GeneratorData Dataset.

        Parameters
        ----------
        smiles: List (required)
            list of smiles to use in dataset

        tokens: list (default None)
            list of characters specifying the language alphabet. Of left
            unspecified, tokens will be extracted from data automatically.

        start_token: str (default '<')
            special character that will be added to the beginning of every
            sequence and encode the sequence start.

        end_token: str (default '>')
            special character that will be added to the end of every
            sequence and encode the sequence end.

        max_len: int (default 120)
            maximum allowed length of the sequences. All sequences longer
            than max_len will be excluded from the training data.
        """

        _BaseChip.__init__(self, tokens=tokens, start_token=start_token, end_token=end_token, pad_token=pad_token,
                           max_len=max_len, use_cuda=use_cuda, device=device)
        Dataset.__init__(self)

        self.vectors = self._vectorify(smiles)

        # remove smis with invalid tokens

        self.num_smiles = len(self.vectors)

    def __len__(self):
        return self.num_smiles

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = self._char_tensor(self.vectors[idx])

        return sample


def _filter_by_dock_scores(smiles_lst, clf, threshold=0.6):
    if len(smiles_lst) == 0:
        return smiles_lst
    _fps, _ = get_fps(smiles_lst, None, func="rdkit")
    if len(_fps.shape) == 1:
        _fps = _fps.reshape(1, -1)

    # sometimes rdkit descriptors fail on valid mols???? This will prevent that
    good_idx = ~np.any(np.isnan(_fps), axis=1)
    dock_scores = np.zeros(len(smiles_lst))

    dock_scores[good_idx] = clf.predict_proba(_fps[good_idx])[:, 1]
    return np.array(smiles_lst)[dock_scores >= threshold]


def _filter_by_diversity(new_smiles, prev_smiles_fps, threshold=1.0):
    if len(new_smiles) == 0:
        return new_smiles
    dists = bulk_fp_tanimoto_similarity(get_fps(new_smiles, func="morgan"), prev_smiles_fps, pooling="max")
    return np.array(new_smiles)[dists < threshold]


def load_pretrained_gen_model(model_path):
    n_src_vocab = len(TOKENS)
    d_word_vec = 512
    n_layers = 8
    n_head = 8
    d_k = 64
    d_inner = 1024

    model = Transformer(n_src_vocab, d_word_vec, n_layers,
                        n_head, d_k, d_inner)

    model.load_state_dict(torch.load(model_path))

    return model


def fine_tune(gen_data: ChipGenerateDataset, model_path: str, use_cuda: bool = True,
              lr: float = 1e-5, n_epochs: int = 10, save_model: str = False):

    model = load_pretrained_gen_model(model_path).to(DEVICE)
    sampler = torch.utils.data.RandomSampler(gen_data)
    train_dataloader = torch.utils.data.DataLoader(gen_data, batch_size=32, shuffle=False, num_workers=0,
                                                   sampler=sampler)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    all_losses = []
    for i in range(n_epochs):
        all_losses += model.fit(loader=train_dataloader, optimizer=optimizer, scheduler=None, n_epochs=1,
                                n_steps=-1, use_cuda=use_cuda, quiet=True, device=DEVICE)
    if save_model:
        torch.save(model.state_dict(), save_model)

    return model


def pretrain(gen_data: ChipGenerateDataset, batch_size: int = 32, lr: float = 0.0001, n_epochs: int = 2,
             use_cuda: bool = True, save_model: str = False):

    model = Transformer(gen_data.n_characters, 512, 8, 8, 64, 1024).to(DEVICE)
    train_dataloader = torch.utils.data.DataLoader(gen_data, batch_size=batch_size, num_workers=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    all_losses = []
    for i in range(n_epochs):
        all_losses += model.fit(loader=train_dataloader, optimizer=optimizer, scheduler=None, n_epochs=1,
                                n_steps=-1, use_cuda=use_cuda, quiet=True, device=DEVICE)
        if save_model:
            torch.save(model.state_dict(), save_model)

    return all_losses, model


def _cluster_biasing_set(smiles, labels, cluster_threshold=0.35):
    clust = cluster_butina(smiles, cluster_threshold)
    top_smiles = []
    top_labels = []
    for c in clust:
        c = np.array(c)
        smiles = smiles[c]
        labels = labels[c]
        top = sorted(list(zip(smiles, labels)), key=lambda x: x[1])[0]
        top_smiles.append(top[0])
        top_labels.append(top[1])
    return top_smiles, top_labels


def generate(model: Transformer,
             use_cuda: bool = True,
             tot_hits: int = 10000,
             batch_size: int = 100,
             save_auxiliary_files: bool = False,
             out_dir: Optional[str] = None,
             prefix: Optional[str] = None):
    denovo_hits = []
    num_hits = 0

    while num_hits < tot_hits:
        batch_denovo = model.generate(batch_size=batch_size, use_cuda=use_cuda, device=DEVICE)
        denovo_hits.extend(batch_denovo)
        num_hits += len(batch_denovo)

        if save_auxiliary_files:
            with open(os.path.join(out_dir, f"{prefix}_denovo_hits.smi"), "w") as f:
                f.write("\n".join(denovo_hits))

    return denovo_hits


def filter_and_generate(smiles: List,
                        labels: List,
                        gen_model_path: str,
                        clf_model_path: Optional[str] = None,
                        do_filter: bool = True,
                        use_cuda: bool = True,
                        tot_hits: int = 10000,
                        batch_size: int = 100,
                        score_quantile: float = 0.01,
                        cluster_biasing_set: bool = True,
                        cluster_threshold: float = 0.35,
                        fine_tune_epochs: int = 10,
                        fine_tune_lr: float = 0.00001,
                        conf_thresh: float = 0.6,
                        diverse_thresh: Optional[float] = None,
                        gen_model_tuned: bool = False,
                        save_models: bool = False,
                        save_auxiliary_files: bool = False,
                        out_dir: Optional[str] = None,
                        prefix: Optional[str] = None
                        ):

    if not use_cuda:
        DEVICE = "cpu"
    else:
        DEVICE = "cuda"

    if prefix is None:
        prefix = str(time.time()).replace('.', '')
    if out_dir is None:
        out_dir = os.getcwd()
    logger.info("starting flatten")
    # flatten and canonicalize smis for CHIP
    smiles, labels = flatten(smiles, labels)
    logger.info("flattened")
    logger.info(f"length smiles: {len(smiles)}")
    logger.info(f"length labels: {len(labels)}")

    # get score threshold (#TODO assumes lower scores are better)
    score_thresh = np.quantile(labels, score_quantile)
    y = (labels <= score_thresh).astype(np.int8)
    logger.info("threshold")
    # train filter model if required
    if do_filter:
        if clf_model_path is None:
            logger.info("starting fps")
            x, y = get_fps(smiles, y, func="rdkit")
            x = x.astype(float)
            logger.info("fps done")
            clf = RandomForestClassifier(n_jobs=-1).fit(x, y)
            logger.info("clf done")
            if save_models:
                dump(clf, os.path.join(out_dir, f"{prefix}_RF.joblib"))
            del x
        else:
            clf = load(clf_model_path)

    smiles = np.array(smiles)
    logger.info("smiles done")
    # collect to smis based on threshold
    top_smiles = smiles[y.astype(bool)]
    top_scores = labels[y.astype(bool)]

    # cluster top set
    if cluster_biasing_set:
        logger.info("I'm clustering")
        top_smiles, top_scores = _cluster_biasing_set(top_smiles, top_scores, cluster_threshold)

    # save additional output files
    if save_auxiliary_files:
        with open(os.path.join(out_dir, f"{prefix}_bias_set.csv"), "w") as f:
            for smi, score in zip(top_smiles, top_scores):
                f.write(f"{smi},{score}\n")

    # finetune generative model if need be
    logger.info("starting finetune")
    fine_tune_dataset = ChipGenerateDataset(top_smiles, use_cuda=use_cuda, device=DEVICE)
    if not gen_model_tuned:
        _save = os.path.join(out_dir, f"{prefix}_tuned.pt") if save_models else False
        model = fine_tune(fine_tune_dataset, model_path=gen_model_path, use_cuda=use_cuda,
                          lr=fine_tune_lr, n_epochs=fine_tune_epochs, save_model=_save)
    else:
        model = load_pretrained_gen_model(gen_model_path)

    logger.info("finetune done")
    del fine_tune_dataset  # remove to save memory

    denovo_hits = []
    denovo_fps = []
    num_hits = 0

    # loop through generation until complete
    while num_hits < tot_hits:
        logger.info("batch")
        batch_denovo_raw = model.generate(batch_size=batch_size, use_cuda=use_cuda, device=DEVICE)
        batch_denovo, _ = flatten(batch_denovo_raw, None)
        batch_hits = np.array(list(set(batch_denovo) - set(top_smiles).union(set(denovo_hits))))
        if do_filter:
            batch_hits = _filter_by_dock_scores(batch_hits, clf, threshold=conf_thresh)
        if diverse_thresh is not None:
            batch_hits = _filter_by_diversity(batch_hits, denovo_fps, threshold=diverse_thresh)
        denovo_hits.extend(batch_hits)
        denovo_fps.extend(get_fps(denovo_hits, None, func="morgan")[0])
        num_hits += len(batch_hits)

        if save_auxiliary_files:
            with open(os.path.join(out_dir, f"{prefix}_denovo_hits.smi"), "w") as f:
                f.write("\n".join(denovo_hits))
            with open(os.path.join(out_dir, f"{prefix}_num_hits.txt"), "a") as f:
                f.write(f"{num_hits}\n")
            with open(os.path.join(out_dir, f"{prefix}_num_hits.txt"), 'r') as f:
                num_epochs = len(f.readlines())
            with open(os.path.join(out_dir, f"{prefix}_status.txt"), 'w') as f:
                f.write(f"Num Epochs: {num_epochs}\nNum Hits: {num_hits}\nLast Update: {str(datetime.now())}\n")

    return denovo_hits


def bias_set_filter_and_generate(bias_smiles,
                                 gen_model_path: str,
                                 clf_model_path: Optional[str] = None,
                                 do_filter: bool = False,
                                 use_cuda: bool = True,
                                 tot_hits: int = 10000,
                                 batch_size: int = 100,
                                 cluster_biasing_set: bool = False,
                                 cluster_threshold: float = 0.35,
                                 fine_tune_epochs: int = 10,
                                 fine_tune_lr: float = 0.0001,
                                 conf_thresh: float = 0.6,
                                 diverse_thresh: Optional[float] = None,
                                 gen_model_tuned: bool = False,
                                 save_models: bool = False,
                                 save_auxiliary_files: bool = False,
                                 out_dir: Optional[str] = None,
                                 prefix: Optional[str] = None
                                 ):

    if not use_cuda:
        DEVICE = "cpu"
    else:
        DEVICE = "cuda"

    if prefix is None:
        prefix = str(time.time()).replace('.', '')
    if out_dir is None:
        out_dir = os.getcwd()

    # flatten and canonicalize smis for CHIP
    bias_smiles = flatten(bias_smiles)

    # train filter model if required
    if do_filter:
        if clf_model_path is None:
            raise ValueError("do_filter cannot be true when clf_model_path is None")
        else:
            clf = load(clf_model_path)

    # cluster top set
    if cluster_biasing_set:
        bias_smiles, _ = _cluster_biasing_set(bias_smiles, [None]*len(bias_smiles),  cluster_threshold)

    # finetune generative model if need be
    fine_tune_dataset = ChipGenerateDataset(bias_smiles, use_cuda=use_cuda, device=DEVICE)
    if not gen_model_tuned:
        _save = os.path.join(out_dir, f"{prefix}_tuned.pt") if save_models else False
        model = fine_tune(fine_tune_dataset, model_path=gen_model_path, use_cuda=use_cuda,
                          lr=fine_tune_lr, n_epochs=fine_tune_epochs, save_model=_save)
    else:
        model = load_pretrained_gen_model(gen_model_path)

    del fine_tune_dataset  # remove to save memory

    denovo_hits = []
    denovo_fps = []
    num_hits = 0

    # loop through generation until complete
    while num_hits < tot_hits:
        batch_denovo_raw = model.generate(batch_size=batch_size, use_cuda=use_cuda, device=DEVICE)
        batch_denovo = flatten(batch_denovo_raw)
        batch_hits = np.array(list(set(batch_denovo) - set(list(bias_smiles)).union(set(denovo_hits))))
        if do_filter:
            batch_hits = _filter_by_dock_scores(batch_hits, clf, threshold=conf_thresh)
        if diverse_thresh is not None:
            batch_hits = _filter_by_diversity(batch_hits, denovo_fps, threshold=diverse_thresh)
        denovo_hits.extend(batch_hits)
        denovo_fps.extend(get_fps(batch_hits, func="morgan"))
        num_hits += len(batch_hits)

        if save_auxiliary_files:
            with open(os.path.join(out_dir, f"{prefix}_denovo_hits.smi"), "w") as f:
                f.write("\n".join(denovo_hits))
            with open(os.path.join(out_dir, f"{prefix}_num_hits.txt"), "a") as f:
                f.write(f"{num_hits}\n")
            with open(os.path.join(out_dir, f"{prefix}_num_hits.txt"), 'r') as f:
                num_epochs = len(f.readlines())
            with open(os.path.join(out_dir, f"{prefix}_status.txt"), 'w') as f:
                f.write(f"Num Epochs: {num_epochs}\nNum Hits: {num_hits}\nLast Update: {str(datetime.now())}\n")

    return denovo_hits


if __name__ == "__main__":
    start_time = time.time()
    parser = argparse.ArgumentParser("chip Fine-Tune and Generate")

    parser.add_argument("--inpath", type=str, required=True,
                        help="Location of docked molecules, with SMILES and Scores.\
                        Expected as a CSV file with at least two columns")
    parser.add_argument("--bias_set", action="store_true", required=False,
                        help="if the inpath dataset is already trimmed to just the bias set. "
                             "Requires pre-generated filter model is filtering is used")
    parser.add_argument("--smi_col", type=str, required=False, default='SMILES',
                        help="Name of SMILES column in input")
    parser.add_argument("--score_col", type=str, default='Score', required=False,
                        help="Name of score column in input")
    parser.add_argument("--outdir", type=str, required=True,
                        help="Where to store saved files. Will create directory if does not exist")
    parser.add_argument("--delimiter", type=str, required=False, default=",",
                        help="column separation char (',' for csv '\\t' for tsv...)")
    parser.add_argument("--prefix", type=str, required=False, default=str(time.time()),
                        help="prefix to attach to all output files")
    parser.add_argument("--save_models", required=False, action="store_true", default=False,
                        help="Whether to save classifier and fine-tuned generator models.\
                        Recommend doing so in order to restart generation after unintentional pause ")
    parser.add_argument("--model_path", type=str, required=False, default=f'{os.path.dirname(__file__)}/models/pretrained_chembl.pt',
                        help="Where to find pre-trained model")
    parser.add_argument('--gen_model_tuned', required=False, action="store_true", default=False,
                        help="generative model is already fine-tuned")
    parser.add_argument("--use_cuda", required=False, action="store_true", default=False,
                        help="Whether to use cuda, only programmed to use cuda:0,\
                        so use CUDA_VISIBLE_DEVICES to set environment variables where desired")
    parser.add_argument("--no_filter", action="store_true", required=False, default=False,
                        help="turn off filtering (faster than setting filter threshold to 0)")
    parser.add_argument("--tot_hits", type=int, required=False, default=10000,
                        help="How many hits to generate. Will continue generating until finished")
    parser.add_argument("--batch_size", type=int, required=False, default=100,
                        help="Batch size for generation. Too large may result in CUDA OOM")
    parser.add_argument("--warm_start", required=False, action="store_true", default=False,
                        help="Whether to start generation from an initial set.\
                        Output of previous generation must be in the same location it was generated in")
    parser.add_argument("--filter_model_path", type=str, required=False, default=None,
                        help="Where to find pre-trained filter model (if one is already trained)")
    parser.add_argument("--conf_threshold", type=float, required=False, default=0.6,
                        help="The screening model confidence threshold to accept a generated chemical")
    parser.add_argument("--cluster_bias", action="store_true", required=False,
                        help="Cluster the biasing set by chemical diversity")
    parser.add_argument("--cluster_thresh", type=float, required=False, default=0.35,
                        help="If clustering bias set, distance threshold to use (higher means fewer clusters)")
    parser.add_argument("--diversity_thresh", type=float, required=False, default=None,
                        help="the maximum tanimoto similarity a new compound can have to another in the generated set")
    parser.add_argument("--score_quantile", type=float, required=False, default=0.01,
                        help="the quantile used to pick the threshold between active and inactive from the given set")
    parser.add_argument("--fine_tune_epochs", type=int, required=False, default=10,
                        help="the number of epochs to do during fine tuning (lower is less biased)")
    parser.add_argument("--fine_tune_lr", type=float, required=False, default=0.00001,
                        help="the learning in rate to use during fine tuning (lower is less biased)")

    args = parser.parse_args()

    if not args.use_cuda:
        DEVICE = "cpu"
    else:
        DEVICE = "cuda"

    if args.bias_set:
        logger.info("starting with bias")
        smiles = load_smiles(args.inpath, delimiter=',', smi_col=args.smi_col, header=True)
        os.makedirs(args.outdir, exist_ok=True)

        if (args.filter_model_path is None) and (not args.no_filter):
            raise ValueError("no_filter cannot be False if not filter model is passed when using manual bias set")

        bias_set_filter_and_generate(smiles,
                                     gen_model_path=args.model_path,
                                     clf_model_path=args.filter_model_path,
                                     do_filter=not args.no_filter,
                                     use_cuda=args.use_cuda,
                                     tot_hits=args.tot_hits,
                                     batch_size=args.batch_size,
                                     cluster_biasing_set=args.cluster_bias,
                                     cluster_threshold=args.cluster_thresh,
                                     fine_tune_epochs=args.fine_tune_epochs,
                                     fine_tune_lr=args.fine_tune_lr,
                                     conf_thresh=args.conf_threshold,
                                     diverse_thresh=args.diversity_thresh,
                                     gen_model_tuned=args.gen_model_tuned,
                                     save_models=args.save_models,
                                     save_auxiliary_files=True,
                                     out_dir=args.outdir,
                                     prefix=args.prefix)

    else:
        logger.info("starting without bias")
        smiles, labels = load_smiles(args.inpath, delimiter=',', smi_col=args.smi_col,
                                     label_col=args.score_col, header=True)
        logger.info("smiles loaded")

        _tuned_model = False
        _filter_model_path = None
        if args.warm_start:
            if not os.path.exists(os.path.join(args.outdir, f"{args.prefix}_tuned.pt")) and \
                    os.path.exists(os.path.join(args.outdir, f"{args.prefix}_status.txt")):
                raise RuntimeError(
                    "cannot warm start without existing fine_tuned and filter models and status in out_dir; "
                    "failed to find models or status file")
            with open(os.path.join(args.outdir, f"{args.prefix}_status.txt"), "r") as _f:
                lines = _f.readlines()
            current_num_hits = int(lines[1].strip().split(":")[-1])

            _tuned_model = True
            args.tot_hits = args.tot_hits - current_num_hits
            args.model_path = os.path.join(args.outdir, f"{args.prefix}_tuned.pt")
            args.filter_model_path = os.path.join(args.outdir, f'{args.prefix}_RF.joblib')

        os.makedirs(args.outdir, exist_ok=True)

        filter_and_generate(smiles=smiles,
                            labels=labels,
                            gen_model_path=args.model_path,
                            clf_model_path=args.filter_model_path,
                            do_filter=not args.no_filter,
                            use_cuda=args.use_cuda,
                            tot_hits=args.tot_hits,
                            batch_size=args.batch_size,
                            score_quantile=args.score_quantile,
                            cluster_biasing_set=args.cluster_bias,
                            cluster_threshold=args.cluster_thresh,
                            fine_tune_epochs=args.fine_tune_epochs,
                            fine_tune_lr=args.fine_tune_lr,
                            conf_thresh=args.conf_threshold,
                            diverse_thresh=args.diversity_thresh,
                            gen_model_tuned=args.gen_model_tuned,
                            save_models=args.save_models,
                            save_auxiliary_files=True,
                            out_dir=args.outdir,
                            prefix=args.prefix)
