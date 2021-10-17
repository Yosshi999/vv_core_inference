from pathlib import Path
from typing import List, Optional

import numpy
import torch
import yaml
from espnet_pytorch_library.tacotron2.decoder import Postnet
from torch import Tensor, nn, onnx
from torch.nn.utils.rnn import pad_sequence
from yukarin_sosoa.config import Config
from yukarin_sosoa.network.predictor import Predictor, create_predictor

from vv_core_inference.utility import remove_weight_norm, to_tensor, OPSET


def make_pad_mask(lengths: Tensor):
    bs = lengths.shape[0]
    maxlen = lengths.max()

    seq_range = torch.arange(0, maxlen, dtype=torch.int64, device=lengths.device)
    seq_range_expand = seq_range.unsqueeze(0).expand(bs, maxlen)
    seq_length_expand = lengths.unsqueeze(-1)
    mask = seq_range_expand >= seq_length_expand
    return mask


def make_non_pad_mask(lengths: Tensor):
    return ~make_pad_mask(lengths)


class WrapperPostnet(nn.Module):
    def __init__(self, net: Postnet):
        super().__init__()
        self.postnet = net.postnet

    def forward(self, xs):
        for net in self.postnet:
            xs = net(xs)
        return xs


class WrapperYukarinSosoa(nn.Module):
    def __init__(self, predictor: Predictor):
        super().__init__()

        self.speaker_embedder = predictor.speaker_embedder
        self.pre = predictor.pre
        self.encoder = predictor.encoder
        self.post = predictor.post
        self.postnet = WrapperPostnet(predictor.postnet)

    @torch.no_grad()
    def forward(
        self,
        f0_pad: Tensor,
        phoneme_pad: Tensor,
        mask: Tensor,
        speaker_id: Tensor,
        speaker_emb: Tensor,
    ):
        h = torch.cat((f0_pad, phoneme_pad), dim=2)  # (batch_size, length, ?)

        print("weight", self.speaker_embedder.num_embeddings)
        speaker_emb[:] = self.speaker_embedder(speaker_id)
        speaker_id = speaker_emb.unsqueeze(dim=1)  # (batch_size, 1, ?)
        speaker_feature = speaker_id.expand(
            speaker_id.shape[0], h.shape[1], speaker_id.shape[2]
        )  # (batch_size, length, ?)
        h = torch.cat((h, speaker_feature), dim=2)  # (batch_size, length, ?)

        h = self.pre(h)

        h, _ = self.encoder(h, mask)

        output1 = self.post(h)
        output2 = output1 + self.postnet(output1.transpose(1, 2)).transpose(1, 2)
        return output2


def make_yukarin_sosoa_forwarder(yukarin_sosoa_model_dir: Path, device, convert=False, onesample=False):
    with yukarin_sosoa_model_dir.joinpath("config.yaml").open() as f:
        config = Config.from_dict(yaml.safe_load(f))

    predictor = create_predictor(config.network)
    state_dict = torch.load(
        yukarin_sosoa_model_dir.joinpath("model.pth"), map_location=device
    )
    predictor.load_state_dict(state_dict)
    predictor.apply(remove_weight_norm)
    predictor.encoder.embed[0].pe = predictor.encoder.embed[0].pe.to(device)
    wrapper = WrapperYukarinSosoa(predictor)
    wrapper.eval().to(device)
    for p in wrapper.parameters():
        p.requires_grad = False
    print("yukarin_sosoa loaded!")

    def _call(
        f0_list: List[Tensor],
        phoneme_list: List[Tensor],
        speaker_id: Optional[numpy.ndarray] = None,
    ):
        if speaker_id is not None:
            speaker_id = to_tensor(speaker_id, device=device)
        length_list = [f0.shape[0] for f0 in f0_list]
        length = torch.tensor(length_list).to(device)

        f0 = pad_sequence(f0_list, batch_first=True)
        phoneme = pad_sequence(phoneme_list, batch_first=True)
        mask = make_non_pad_mask(length).to(device).unsqueeze(-2)
        args = (
            f0,
            phoneme,
            mask,
            speaker_id,
            torch.zeros(f0.shape[0], predictor.speaker_embedder.embedding_dim, device=device)
        )
        output = wrapper(*args)
        if convert:
            onnx.export(
                wrapper,
                args,
                "yukarin_sosoa.onnx",
                opset_version=OPSET,
                do_constant_folding=True,
                input_names=[
                    "f0_pad",
                    "phoneme_pad",
                    "mask",
                    "speaker_id",
                    "speaker_emb",
                ],
                output_names=["spec"],
                dynamic_axes={
                    "f0_pad": {0: "batch", 1: "padded_length"},
                    "phoneme_pad": {0: "batch", 1: "padded_length", 2: "phoneme_size"},
                    "mask": {2: "padded_length"},
                    "speaker_emb": {0: "batch", 1: "speaker_embedding"},
                    "spec": {0: "batch", 1: "padded_length"},
                },
                example_outputs=output)
            print("yukarin_sosoa has been converted to ONNX")
        return [output[i, :l] for i, l in enumerate(length_list)]

    def _call_onesample(
        f0: Tensor,
        phoneme: Tensor,
        speaker_id: Tensor,
    ):
        mask = torch.ones_like(f0, dtype=torch.bool).squeeze()
        f0 = f0.unsqueeze(0)
        phoneme = phoneme.unsqueeze(0)
        speaker_emb = torch.zeros(1, predictor.speaker_embedder.embedding_dim, device=device)
        return wrapper(f0, phoneme, mask, speaker_id, speaker_emb)

    return _call_onesample if onesample else _call
