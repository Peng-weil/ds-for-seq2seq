import argparse
import math
import torch
from torch.utils.data import DataLoader

import src.utils
from src.model import build_modules
from src.utils import bool_flag, init_exp, to_cuda, mkdir

from src.data_loader import Matrix_Dataset


def testNN(modules, testLoader):
    with torch.no_grad():
        encoder, decoder = modules['encoder'], modules['decoder']
        encoder.eval()
        decoder.eval()

        # stats
        xe_loss = 0

        num_single = 0
        correct_num_single = 0

        correct_num_total = 0

        # test
        iterator = enumerate(testLoader)

        for _, data in iterator:
            (x1, len1), (x2, len2) = data

            alen = torch.arange(len2.max(), dtype=torch.long, device=len2.device)
            pred_mask = alen[:, None] < len2[None] - 1  # do not predict anything given the last target word
            y = x2[1:].masked_select(pred_mask[:-1])
            assert len(y) == (len2 - 1).sum().item()
            # cuda
            x1, len1, x2, len2, y = to_cuda(x1, len1, x2, len2, y)
            # forward / loss
            encoded = encoder('fwd', x=x1, lengths=len1, causal=False)
            decoded = decoder('fwd', x=x2, lengths=len2, causal=True, src_enc=encoded.transpose(0, 1), src_len=len1)
            word_scores, loss = decoder('predict', tensor=decoded, pred_mask=pred_mask, y=y, get_scores=True)

            t = torch.zeros_like(pred_mask, device=y.device)
            pred_res = word_scores.max(1)[1]
            t[pred_mask] += pred_res == y
            # res = (t.T)
            res = (t.T)[:, 0:-2]

            # xe_loss
            xe_loss += loss.item() * len(y)

            # single accuracy
            num_single += res.numel()
            correct_num_single += (res == True).cpu().long().sum().item()

            # total accuracy
            correct_num_total += (res.sum(1) == len2 - 2).cpu().long().sum().item()

        acc_single = (correct_num_single / num_single) * 100
        acc_total = (correct_num_total / len(testLoader.dataset)) * 100

    return xe_loss / len(testLoader.dataset), math.trunc(acc_single * 100) / 100, math.trunc(acc_total * 100) / 100


def main(params):
    logger = init_exp(params)

    if not params.cpu:
        assert torch.cuda.is_available()
    src.utils.CUDA = not params.cpu
    device = torch.device('cuda' if (torch.cuda.is_available() and not params.cpu) else 'cpu')
    logger.info(f'device is {device.type}')

    # env
    params.words = ['<s>', '</s>', '<pad>', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    params.id2word = {i: s for i, s in enumerate(params.words)}
    params.word2id = {s: i for i, s in params.id2word.items()}
    params.n_words = len(params.words)
    params.eos_index = 0
    params.pad_index = 1
    logger.info(f"word2id: {params.word2id}")

    modules = build_modules(params)

    test_loader_dict = {}
    for k, v in params.reload_testset.items():
        test_dataset = Matrix_Dataset(v, train=False, params=params)
        test_loader_dict[k] = DataLoader(test_dataset, batch_size=32, shuffle=True, collate_fn=test_dataset.collate_fn)

    for ds_type, loader in test_loader_dict.items():
        xe_loss, acc_single, acc_total = testNN(modules, loader)
        logger.info(
            f"{ds_type:<10} - xe_loss: {xe_loss}, acc_single:{acc_single:.2f} %, acc_total: {acc_total:.2f} %")


def get_parser():
    """
    Generate a parameters parser.
    """
    # parse parameters
    parser = argparse.ArgumentParser(description="test")

    # main parameters
    parser.add_argument('--dump_path', type=str, default='./dumped/')
    parser.add_argument('--exp_name', type=str, default='transpose_test')
    parser.add_argument('--exp_id', type=str, default='')
    parser.add_argument("--batch_size", type=int, default=16)

    parser.add_argument("--env_base_seed", type=int, default=0)
    parser.add_argument('--num_workers', type=int, default=0)

    parser.add_argument('--cpu', type=bool_flag, default=False)

    # model parameters
    parser.add_argument('--emb_dim', type=int, default=256)
    parser.add_argument('--n_enc_layers', type=int, default=6)
    parser.add_argument('--n_dec_layers', type=int, default=6)
    parser.add_argument('--n_heads', type=int, default=8)
    parser.add_argument('--dropout', type=float, default=0)
    parser.add_argument('--attention_dropout', type=float, default=0)
    parser.add_argument('--share_inout_emb', type=bool_flag, default=True)
    parser.add_argument('--sinusoidal_embeddings', type=bool_flag, default=False)

    parser.add_argument('--reload_data', type=str,
                        default='./dataset/TRA/1000_10_10_24eax')
    parser.add_argument('--reload_size', type=int, default=-1)
    parser.add_argument("--reload_model", type=str,
                        default="./dumped/transpose_test/5gc5qcc6oh/best_total_acc.pth")
    parser.add_argument("--reload_checkpoint", type=str, default="")
    parser.add_argument('--reload_testset', type=dict,
                        default={
                            "rank_10_10": "./dataset/TRA/1000_10_10_24eax",
                            "rank_9_9": "./dataset/TRA/1000_9_9_37fl9",
                            "rank_8_8": "./dataset/TRA/1000_8_8_ezs80",
                            "rank_7_7": "./dataset/TRA/1000_7_7_3iz55",
                            "rank_6_6": "./dataset/TRA/1000_6_6_in1dd"})

    return parser


if __name__ == '__main__':
    parser = get_parser()
    params = parser.parse_args()

    main(params)
