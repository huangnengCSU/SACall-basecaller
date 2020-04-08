import torch
import torch.nn as nn
from torch.autograd import Variable

from transformer.modules import Transformer
from transformer.Beam import Beam


class Decode(object):
    ''' Load with trained model and handle the beam search '''

    def __init__(self, opt, device):
        self.opt = opt
        self.device = device

        checkpoint = torch.load(opt.model)
        model_opt = checkpoint['settings']
        self.model_opt = model_opt

        model = Transformer(d_model=model_opt.d_model,
                            d_ff=model_opt.d_ff,
                            n_head=model_opt.n_head,
                            num_encoder_layers=model_opt.encoder_layers,
                            num_decoder_layers=model_opt.decoder_layers,
                            label_vocab_size=model_opt.label_vocab_size,
                            d_word_vec=model_opt.d_word_vec,
                            dropout=model_opt.dropout).to(device)

        model.load_state_dict(checkpoint['model'])
        print('[Info] Trained model state loaded.')

        self.model = model
        self.model.eval()

    def decode_batch(self, src_batch):
        ''' Translation work in one batch '''

        # Batch size is in different location depending on data.
        src_seq = src_batch
        batch_size = src_seq.size(0)
        beam_size = self.opt.beam_size

        # - Enocde
        enc_output, src_mask = self.model.encoder(src_seq)
        # print('enc_output.size', enc_output.size())     #(batch, length, d_model)
        # print('src_mask.size()', src_mask.size())  # (batch, 1, length)


        # (batch * beam_size, length, d_model)
        enc_output = Variable(
            enc_output.data.repeat(1, beam_size, 1).view(
                enc_output.size(0) * beam_size, enc_output.size(1), enc_output.size(2)))

        # (batch * beam_size, 1, d_model)
        src_mask = src_mask.repeat(1, beam_size, 1).view(
            src_mask.size(0) * beam_size, src_mask.size(1), src_mask.size(2))



        # --- Prepare beams
        beams = [Beam(beam_size, self.device) for _ in range(batch_size)]
        # print('beams:',beams)
        beam_inst_idx_map = {
            beam_idx: inst_idx for inst_idx, beam_idx in enumerate(range(batch_size))}
        # print('beam_inst_idx_map:',beam_inst_idx_map)
        n_remaining_sents = batch_size

        # - Decode
        for i in range(self.model_opt.label_max_len):

            # print('-'*20)
            len_dec_seq = i + 1
            # print(len_dec_seq)

            # -- Preparing decoded data seq -- #
            # size: (batch , beam , len_dec_seq)
            dec_partial_seq = torch.stack([
                b.get_current_state() for b in beams if not b.done])
            # print('dec_partial_seq 1',dec_partial_seq.size())

            # size: (batch * beam , len_dec_seq)
            dec_partial_seq = dec_partial_seq.view(-1, len_dec_seq)
            # print('dec_partial_seq 2:\n', dec_partial_seq)

            dec_partial_seq = dec_partial_seq.to(self.device)



            # -- Decoding -- #

            # (batch * beam, len_dec_seq, d_model)
            dec_output = self.model.decoder(dec_partial_seq, enc_output, src_mask)
            # print('dec_output:',dec_output.size())

            # (batch * beam, d_model)
            dec_output = dec_output[:, -1, :]

            # (batch * beam, vocab_size)
            dec_output = self.model.final_proj(dec_output)
            # print('decoder output shape:', dec_output.size())

            # (batch * beam, vocab_size) logSoftmax
            out = self.model.log_softmax(dec_output)

            # (batch , beam , vocab_size)
            word_lk = out.view(n_remaining_sents, beam_size, -1).contiguous()

            active_beam_idx_list = []

            for beam_idx in range(batch_size):
                # current case in batch, is predicted EOS.
                if beams[beam_idx].done:
                    # print('continue','\n'*100)
                    continue

                inst_idx = beam_inst_idx_map[beam_idx]

                # print('word_lk.data[%d]'%(inst_idx),word_lk.data[inst_idx])

                # word_lk.data[inst_idx] (beam_size, vocab_size) current inst of batch
                if not beams[beam_idx].advance(word_lk.data[inst_idx]):
                    active_beam_idx_list += [beam_idx]

            if not active_beam_idx_list:
                # all instances have finished their path to <EOS>
                break

            # in this section, the sentences that are still active are
            # compacted so that the decoder is not run on completed sentences
            active_inst_idxs = torch.LongTensor(
                [beam_inst_idx_map[k] for k in active_beam_idx_list])

            # update the idx mapping
            beam_inst_idx_map = {
                beam_idx: inst_idx for inst_idx, beam_idx in enumerate(active_beam_idx_list)}
            # print('beam_inst_idx_map2:\n',beam_inst_idx_map)

            def update_active_seq(seq_var, active_inst_idxs):
                ''' Remove the src sequence of finished instances in one batch. '''

                inst_idx_dim_size, *rest_dim_sizes = seq_var.size()
                inst_idx_dim_size = inst_idx_dim_size * \
                                    len(active_inst_idxs) // n_remaining_sents
                new_size = (inst_idx_dim_size, *rest_dim_sizes)

                # select the active instances in batch
                original_seq_data = seq_var.data.view(n_remaining_sents, -1)
                active_seq_data = original_seq_data.index_select(
                    0, active_inst_idxs)
                active_seq_data = active_seq_data.view(*new_size)
                with torch.no_grad():
                    return Variable(active_seq_data)

            def update_active_enc_info(enc_info_var, active_inst_idxs):
                ''' Remove the encoder outputs of finished instances in one batch. '''

                # (batch * beam, length, d_model)
                inst_idx_dim_size, *rest_dim_sizes = enc_info_var.size()

                inst_idx_dim_size = inst_idx_dim_size * len(active_inst_idxs) // n_remaining_sents
                new_size = (inst_idx_dim_size, *rest_dim_sizes)
                # print('new_size:\n',new_size)
                # print(n_remaining_sents)

                # select the active instances in batch
                # (batch, beam, d_model)
                original_enc_info_data = enc_info_var.data.view(n_remaining_sents, -1, enc_info_var.size(2))

                # select instance of batch (new_batch, beam, d_model)
                active_enc_info_data = original_enc_info_data.index_select(0, active_inst_idxs)
                active_enc_info_data = active_enc_info_data.view(*new_size)
                with torch.no_grad():
                    return Variable(active_enc_info_data)

            enc_output = update_active_enc_info(enc_output, active_inst_idxs.to(self.device))
            src_mask = update_active_enc_info(src_mask,active_inst_idxs.to(self.device))

            # - update the remaining size
            n_remaining_sents = len(active_inst_idxs)

        # - Return useful information
        all_hyp, all_scores = [], []
        n_best = self.opt.n_best

        for beam_idx in range(batch_size):
            scores, tail_idxs = beams[beam_idx].sort_scores()
            all_scores += [scores[:n_best]]

            # hyps1 = [beams[beam_idx].get_hypothesis(
            #     i) for i in tail_idxs[:n_best]]
            # print(torch.LongTensor(hyps1))
            hyps = torch.LongTensor(beams[beam_idx].bestpath)[:n_best,1:]
            # print(hyps)
            # assert torch.LongTensor(hyps1).equal(hyps)
            all_hyp += [hyps]

        return all_hyp, all_scores
