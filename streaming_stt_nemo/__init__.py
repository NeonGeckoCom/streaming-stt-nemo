import numpy as np
import resampy

import ctypes, gc

import torch
import sentencepiece as spm
import onnxruntime as ort

from huggingface_hub import hf_hub_download

from .configs import languages, sample_rate, subfolder_name


class Model:
    langs = languages
    sample_rate = sample_rate


    def __init__(self, lang="en"):
        self._init_model(lang)
        
    def _init_model(self, lang: str):
        model_name = self.langs[lang]["model"]
        self._init_preprocessor(model_name)
        self._init_encoder(model_name)
        self._init_tokenizer(model_name)

    def _init_preprocessor(self, model_name: str):
        preprocessor_path = hf_hub_download(model_name, "preprocessor.ts", subfolder=subfolder_name)
        self.preprocessor = torch.jit.load(preprocessor_path)

    def _init_encoder(self, model_name: str):
        encoder_path = hf_hub_download(model_name, "model.onnx", subfolder=subfolder_name)
        self.encoder = ort.InferenceSession(encoder_path)

    def _init_tokenizer(self, model_name: str):
        tokenizer_path = hf_hub_download(model_name, "tokenizer.spm", subfolder=subfolder_name)
        self.tokenizer = spm.SentencePieceProcessor(tokenizer_path)

    def _run_preprocessor(self, audio_16k: np.array):
        processed_signal, processed_signal_len = self.preprocessor.forward(
            input_signal = torch.tensor([audio_16k]), length = torch.tensor([len(audio_16k)])
        )
        processed_signal = processed_signal.numpy()
        processed_signal_len = processed_signal_len.numpy()
        return processed_signal, processed_signal_len

    def _run_encoder(self, processed_signal: np.array, processed_signal_len: np.array):
        outputs = self.encoder.run(None, {'audio_signal': processed_signal,
                              'length':processed_signal_len})
        logits = outputs[0][0]
        return logits

    def _run_tokenizer(self, logits: np.array):
        blank_id = self.tokenizer.vocab_size()
        decoded_prediction = self._ctc_decode(logits, blank_id)
        text = self.tokenizer.decode_ids(decoded_prediction)
        current_hypotheses = [text]
        return current_hypotheses

    @staticmethod
    def _ctc_decode(logits: np.array, blank_id: int):
        labels = logits.argmax(axis=1).tolist()

        previous = blank_id
        decoded_prediction = []
        for p in labels:
            if (p != previous or previous == blank_id) and p != blank_id:
                decoded_prediction.append(p)
            previous = p
        return decoded_prediction
        

    def stt(self, audio_buffer: np.array, sr: int):
        audio_fp32 = self._to_float32(audio_buffer)
        audio_16k = self._resample(audio_fp32, sr)

        processed_signal, processed_signal_len = self._run_preprocessor(audio_16k)
        logits = self._run_encoder(processed_signal, processed_signal_len)
        current_hypotheses = self._run_tokenizer(logits)

        return current_hypotheses
    

    def _resample(self, audio_fp32: np.array, sr: int):
        audio_16k = resampy.resample(audio_fp32, sr, self.sample_rate)
        return audio_16k


    def _to_float32(self, audio_buffer: np.array):
        audio_fp32 = np.divide(audio_buffer, np.iinfo(audio_buffer.dtype).max, dtype=np.float32)
        return audio_fp32