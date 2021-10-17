from pathlib import Path
from typing import Optional

import torch
import yaml
from torch import Tensor, nn, onnx
from yukarin_sa.config import Config
from yukarin_sa.network.predictor import Predictor, create_predictor

from vv_core_inference.utility import remove_weight_norm, to_tensor, OPSET


class WrapperUniGRU(nn.Module):
    def __init__(self, rnn: nn.GRU):
        super().__init__()
        self.rnn = rnn

    def forward(self, x: Tensor, hidden: Tensor):
        output, hidden = self.rnn(x.transpose(1, 2), hidden)
        return output.transpose(1, 2), hidden


class WrapperYukarinSa(nn.Module):
    def __init__(self, predictor: Predictor):
        super().__init__()
        self.phoneme_embedder = predictor.phoneme_embedder
        self.speaker_embedder = predictor.speaker_embedder
        self.encoder = predictor.encoder
        self.ar_encoder = WrapperUniGRU(predictor.ar_encoder.rnn)
        self.post = predictor.post

    @torch.no_grad()
    def forward(
        self,
        length: Tensor,
        vowel_phoneme_list: Tensor,
        consonant_phoneme_list: Tensor,
        start_accent_list: Tensor,
        end_accent_list: Tensor,
        start_accent_phrase_list: Tensor,
        end_accent_phrase_list: Tensor,
        speaker_id: Tensor,
        # placeholders below #
        ph: Tensor,
        speaker_emb: Tensor,
        encoder_input: Tensor,
        ar_encoder_input: Tensor,
        ar_encoder_hidden: Tensor,
    ):
        batch_size = vowel_phoneme_list.shape[0]

        ph[:] = self.phoneme_embedder(vowel_phoneme_list + 1) + self.phoneme_embedder(
            consonant_phoneme_list + 1
        )  # (batch_size, length, _phenome_emb)
        ph = ph.transpose(1, 2)  # (batch_size, _phenome_emb, length)

        ah = torch.stack(
            [
                start_accent_list,
                end_accent_list,
                start_accent_phrase_list,
                end_accent_phrase_list,
            ],
            dim=1,
        ).to(
            ph.dtype
        )  # (batch_size, 4, length)

        _h = torch.cat((ph, ah), dim=1)  # (batch_size, _phenome_emb + 4, length)

        speaker_emb[:] = self.speaker_embedder(speaker_id)  # (batch_size, _speaker_emb)
        speaker_id = speaker_emb.unsqueeze(2)  # (batch_size, _speaker_emb, 1)
        speaker = speaker_id.expand(
            speaker_id.shape[0], speaker_id.shape[1], ph.shape[2]
        )  # (batch_size, _speaker_emb, length)
        encoder_input[:] = torch.cat((_h, speaker), dim=1)  # (batch_size, encoder_emb = _phoneme_emb + 4 + speaker_emb, length)

        h = self.encoder(encoder_input)  # (batch_size, encoder_emb, length)

        f0_one = torch.zeros(
            batch_size, 1, 1, dtype=h.dtype, device=h.device
        )  # (batch_size, 1, 1)

        hidden = ar_encoder_hidden
        f0 = []
        for i in range(int(length)):
            h_one = h[:, :, i : i + 1]  # (batch_size, encoder_emb, 1)
            ar_encoder_input[:] = torch.cat((h_one, f0_one), dim=1)  # (batch_size, encoder_emb+1, 1)
            h_one, hidden = self.ar_encoder(
                ar_encoder_input, hidden=hidden
            )  # (batch_size, ?, 1)
            f0_one = self.post(h_one)  # (batch_size, 1, 1)

            f0 += [f0_one[:, 0, 0]]
        return torch.stack(f0, dim=1) # (batch_size, length)

def make_yukarin_sa_forwarder(yukarin_sa_model_dir: Path, device, convert=False):
    with yukarin_sa_model_dir.joinpath("config.yaml").open() as f:
        config = Config.from_dict(yaml.safe_load(f))

    predictor = create_predictor(config.network)
    state_dict = torch.load(
        yukarin_sa_model_dir.joinpath("model.pth"), map_location=device
    )
    predictor.load_state_dict(state_dict)
    predictor.apply(remove_weight_norm)
    wrapper = WrapperYukarinSa(predictor)
    wrapper.eval().to(device)
    print("yukarin_sa loaded!")

    def _call(
        length: int,
        vowel_phoneme_list: Tensor,
        consonant_phoneme_list: Tensor,
        start_accent_list: Tensor,
        end_accent_list: Tensor,
        start_accent_phrase_list: Tensor,
        end_accent_phrase_list: Tensor,
        speaker_id: Optional[Tensor],
    ):
        length = to_tensor(length, device=device)
        vowel_phoneme_list = to_tensor(vowel_phoneme_list, device=device)
        consonant_phoneme_list = to_tensor(consonant_phoneme_list, device=device)
        start_accent_list = to_tensor(start_accent_list, device=device)
        end_accent_list = to_tensor(end_accent_list, device=device)
        start_accent_phrase_list = to_tensor(
            start_accent_phrase_list, device=device
        )
        end_accent_phrase_list = to_tensor(end_accent_phrase_list, device=device)

        if speaker_id is not None:
            speaker_id = to_tensor(speaker_id, device=device)
            speaker_id = speaker_id.reshape((-1,)).to(torch.int64)

        batch_size, length_size = vowel_phoneme_list.shape[0], vowel_phoneme_list.shape[1]
        phoneme_emb_dim = predictor.phoneme_embedder.embedding_dim
        speaker_emb_dim = predictor.speaker_embedder.embedding_dim
        encoder_emb_dim = phoneme_emb_dim + 4 + speaker_emb_dim

        _rnn = predictor.ar_encoder.rnn
        num_directions = 2 if _rnn.bidirectional else 1
        args = (
            length,
            vowel_phoneme_list,
            consonant_phoneme_list,
            start_accent_list,
            end_accent_list,
            start_accent_phrase_list,
            end_accent_phrase_list,
            speaker_id,
            torch.zeros(batch_size, length_size, phoneme_emb_dim, device=device),
            torch.zeros(batch_size, speaker_emb_dim, device=device),
            torch.zeros(batch_size, encoder_emb_dim, length_size, device=device),
            torch.zeros(batch_size, encoder_emb_dim+1, 1, device=device),
            torch.zeros(_rnn.num_layers * num_directions,
                        batch_size, _rnn.hidden_size,
                        device=device),
        )
        output = wrapper(*args)
        if convert:
            onnx.export(
                torch.jit.script(wrapper),
                args,
                "yukarin_sa.onnx",
                opset_version=OPSET,
                do_constant_folding=True,
                input_names=[
                    "length",
                    "vowel_phoneme_list",
                    "consonant_phoneme_list",
                    "start_accent_list",
                    "end_accent_list",
                    "start_accent_phrase_list",
                    "end_accent_phrase_list",
                    "speaker_id",
                    "ph",
                    "speaker_emb",
                    "encoder_input",
                    "ar_encoder_input",
                    "ar_encoder_hidden",
                ],
                output_names=["f0_list"],
                dynamic_axes={
                    "vowel_phoneme_list": {0: "batch", 1: "length"},
                    "consonant_phoneme_list": {0: "batch", 1: "length"},
                    "start_accent_list": {0: "batch", 1: "length"},
                    "end_accent_list": {0: "batch", 1: "length"},
                    "start_accent_phrase_list": {0: "batch", 1: "length"},
                    "end_accent_phrase_list": {0: "batch", 1: "length"},
                    "ph": {0: "batch", 1: "length", 2: "phoneme_embedding"},
                    "speaker_emb": {0: "batch", 1: "speaker_embedding"},
                    "encoder_input": {0: "batch", 1: "encoder_embedding", 2: "length"},
                    "ar_encoder_input": {0: "batch"},
                    "ar_encoder_hidden": {1: "batch"},
                    "f0_list": {0: "batch", 1: "phoneme_embedding", 2: "length"}},
                example_outputs=output)
            print("yukarin_sa has been converted to ONNX")
        return output.cpu().numpy()

    return _call
