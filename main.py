# PYTHONUNBUFFERED=1;CUDA_VISIBLE_DEVICES=0

import argparse
import os
import math

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_

import src.utils
from src.model import build_modules
from src.utils import bool_flag, init_exp, to_cuda, mkdir

from src.data_loader import Matrix_Dataset

np.seterr(all='raise')


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


def save_checkpint(path, params, epoch, total_iter, best_total_acc, optimizer, modules):
    data = {
        'epoch': epoch,
        'total_iter': total_iter,
        'best_metrics': best_total_acc,
        'params': {k: v
                   for k, v in params.__dict__.items()},
    }

    for k, v in modules.items():
        data[k] = v.state_dict()

    data[f'optimizer'] = optimizer.state_dict()
    torch.save(data, path)


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

    # dataloader
    data_pth = params.reload_data

    matrix_dataset = Matrix_Dataset(data_pth, train=True, params=params)

    train_size = int(0.8 * len(matrix_dataset))
    test_size = int(0.2 * len(matrix_dataset))
    trainDataset, testDataset = torch.utils.data.random_split(matrix_dataset, [train_size, test_size])

    trainLoader = DataLoader(trainDataset,
                             batch_size=params.batch_size,
                             num_workers=(params.num_workers if data_pth is None or params.num_workers == 0 else 1),
                             shuffle=True,
                             collate_fn=matrix_dataset.collate_fn)

    trainAccLoader = DataLoader(trainDataset, batch_size=32, shuffle=True, collate_fn=matrix_dataset.collate_fn)
    testAccLoader = DataLoader(testDataset, batch_size=32, shuffle=False, collate_fn=matrix_dataset.collate_fn)

    # Initialize NN model, define optimizer
    # model
    modules = build_modules(params)

    named_parameters = []
    for v in modules.values():
        named_parameters.extend([(k, p) for k, p in v.named_parameters() if p.requires_grad])
    model_parameters = {'model': [p for k, p in named_parameters]}
    for k, v in model_parameters.items():
        logger.info("Found %i parameters in %s." % (len(v), k))
        assert len(v) >= 1

    optimizer = torch.optim.Adam(model_parameters['model'], lr=0.0001, betas=(0.9, 0.999))

    total_iter = 0

    best_total_acc = -1.0
    stop_iter = 0

    # model save path
    dir_checkpoint = os.path.join(params.dump_path, 'checkpoint')
    mkdir(dir_checkpoint)

    for epoch in range(params.max_epoch):

        logger.info("\n============ Starting epoch %i ... ============" % epoch)

        encoder, decoder = modules['encoder'], modules['decoder']
        encoder.train()
        decoder.train()

        iterator = enumerate(trainLoader)
        for _, data in iterator:
            (x1, len1), (x2, len2) = data

            # target words to predict
            alen = torch.arange(len2.max(), dtype=torch.long, device=len2.device)
            pred_mask = alen[:, None] < len2[None] - 1  # do not predict anything given the last target word
            y = x2[1:].masked_select(pred_mask[:-1])
            assert len(y) == (len2 - 1).sum().item()

            # cuda
            x1, len1, x2, len2, y = to_cuda(x1, len1, x2, len2, y)

            # forward / loss
            encoded = encoder('fwd', x=x1, lengths=len1, causal=False)
            decoded = decoder('fwd', x=x2, lengths=len2, causal=True, src_enc=encoded.transpose(0, 1), src_len=len1)
            _, loss = decoder('predict', tensor=decoded, pred_mask=pred_mask, y=y, get_scores=False)

            # check NaN
            if (loss != loss).data.any():
                logger.warning("NaN detected")

            optimizer.zero_grad()
            loss.backward()
            if params.clip_grad_norm > 0:
                clip_grad_norm_(model_parameters['model'], params.clip_grad_norm)
            optimizer.step()

            # print
            total_iter += 1

            if total_iter % 20 != 0:
                continue

            s_iter = "%7i - " % total_iter

            s_stat = "{}: {:.2f}".format("cross_entropy", loss.item())

            for group in optimizer.param_groups:
                s_lr = "LR: {:.4e}".format(group['lr'])

            logger.info(f"{s_iter} {s_stat} {s_lr}")

        # test NNs in test set
        logger.info("*" * 16 + "Test of epoch %i :" % epoch + "*" * 16)
        xe_loss_trainset, acc_single_trainset, acc_total_trainset = testNN(modules, trainAccLoader)
        logger.info(
            f"{'trainset':<10} - xe_loss: {xe_loss_trainset}, acc_single:{acc_single_trainset:.2f} %, acc_total: {acc_total_trainset:.2f} %")

        xe_loss_testset, acc_single_testset, acc_total_testset = testNN(modules, testAccLoader)
        logger.info(
            f"{'testset':<10} - xe_loss: {xe_loss_testset}, acc_single:{acc_single_testset:.2f} %, acc_total: {acc_total_testset:.2f} %")

        if acc_total_testset > best_total_acc:
            stop_iter = 0
            best_total_acc = acc_total_testset

            # save best model
            logger.info(f'Best \"test_total_acc\" update for: {best_total_acc:.2f} %')
            pth_best_model = os.path.join(params.dump_path, "best_total_acc.pth")
            logger.info(f"Saving the best model in {pth_best_model}")
            save_checkpint(path=pth_best_model,
                           params=params,
                           epoch=epoch,
                           total_iter=total_iter,
                           best_total_acc=best_total_acc,
                           optimizer=optimizer,
                           modules=modules)

        else:
            if acc_total_testset > 0.0:
                stop_iter += 1
            logger.info(
                f"No better \"test_total_acc\" has been achieved for {stop_iter} / {params.stopping_criterion_acc} th of the rounds,\n"
                f"and the optimal result is currently {best_total_acc} %"
            )

        # save model
        if epoch % 50 == 0:
            pth_checkpoint = os.path.join(dir_checkpoint, "checkpoint_epoch" + str(epoch) + ".pth")
            logger.info(f"Saving the checkpoint model in {pth_checkpoint}")

            save_checkpint(path=pth_checkpoint,
                           params=params,
                           epoch=epoch,
                           total_iter=total_iter,
                           best_total_acc=best_total_acc,
                           optimizer=optimizer,
                           modules=modules)

        logger.info("============ End of epoch %i ============" % epoch)
        if not stop_iter < params.stopping_criterion_acc:
            break

        if acc_total_trainset == 100.0 and acc_total_testset == 100.0:
            logger.info("Evaluation metric achieved 100.00 % with early exit.")
            break

    # test model
    logger.info("\n" + "*" * 16 + "Testing on a matrix dataset of additional ranks" + "*" * 16)
    test_loader_dict = {}
    for k, v in params.reload_testset.items():
        test_dataset = Matrix_Dataset(v, train=False, params=params)
        test_loader_dict[k] = DataLoader(test_dataset, batch_size=32, shuffle=True, collate_fn=test_dataset.collate_fn)

    for ds_type, loader in test_loader_dict.items():
        xe_loss, acc_single, acc_total = testNN(modules, loader)
        logger.info(
            f"{ds_type:<10} - xe_loss: {xe_loss}, acc_single:{acc_single:.2f} %, acc_total: {acc_total:.2f} %")

    logger.info("Running complete.")


def get_parser():
    """
    Generate a parameters parser.
    """
    # parse parameters
    parser = argparse.ArgumentParser(description="Matrix")

    # main parameters
    parser.add_argument('--dump_path', type=str, default='./dumped/')
    parser.add_argument('--exp_name', type=str, default='transpose')
    parser.add_argument('--exp_id', type=str, default='')

    # training parameters
    parser.add_argument("--env_base_seed", type=int, default=0)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_epoch", type=int, default=10000)
    parser.add_argument("--clip_grad_norm", type=float, default=5)
    parser.add_argument("--stopping_criterion_acc", type=int, default=100)
    parser.add_argument('--cpu', type=bool_flag, default=False)
    parser.add_argument("--sort_method", type=str, default="SMD,SMR")

    # model parameters
    parser.add_argument('--emb_dim', type=int, default=256)
    parser.add_argument('--n_enc_layers', type=int, default=6)
    parser.add_argument('--n_dec_layers', type=int, default=6)
    parser.add_argument('--n_heads', type=int, default=8)
    parser.add_argument('--dropout', type=float, default=0)
    parser.add_argument('--attention_dropout', type=float, default=0)
    parser.add_argument('--share_inout_emb', type=bool_flag, default=True)
    parser.add_argument('--sinusoidal_embeddings', type=bool_flag, default=False)

    # reload pretrained model / checkpoint
    parser.add_argument('--reload_data', type=str,
                        default='./dataset/MIS/10000_20_20_645qa')
    parser.add_argument('--reload_size', type=int, default=-1)
    parser.add_argument("--reload_model", type=str, default="")
    parser.add_argument("--reload_checkpoint", type=str, default="")
    parser.add_argument('--reload_testset', type=dict,
                        default={
                            '20 nodes': './dataset/MIS/10000_20_20_645qa',
                            '19 nodes': './dataset/MIS/10000_19_19_3gsaa',
                            '18 nodes': './dataset/MIS/10000_18_18_0p7ku',
                            '17 nodes': './dataset/MIS/10000_17_17_ejchv',
                            '16 nodes': './dataset/MIS/10000_16_16_37tz5'})

    return parser


if __name__ == '__main__':
    parser = get_parser()
    params = parser.parse_args()

    main(params)
