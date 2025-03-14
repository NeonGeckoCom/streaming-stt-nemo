# NEON AI (TM) SOFTWARE, Software Development Kit & Application Framework
# All trademark and other rights reserved by their respective owners
# Copyright 2008-2025 Neongecko.com Inc.
# Contributors: Daniel McKnight, Guy Daniels, Elon Gasper, Richard Leeds,
# Regina Bloomstine, Casimiro Ferreira, Andrii Pernatii, Kirill Hrymailo
# BSD-3 License
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# 1. Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from this
#    software without specific prior written permission.
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
# THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
# CONTRIBUTORS  BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA,
# OR PROFITS;  OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
# LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
# NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE,  EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import ctypes
import gc
import os.path

import numpy as np
import onnxruntime as ort
import sentencepiece as spm
import soxr
import torch
from huggingface_hub import hf_hub_download
from pydub import AudioSegment

from .configs import languages, sample_rate, subfolder_name

available_languages = list(languages.keys())


class Model:
    langs = languages
    sample_rate = sample_rate

    def __init__(self, lang="en", model_folder=None):
        if model_folder:
            self._init_model_from_path(model_folder)
        else:
            self._init_model(lang)

    def _init_model(self, lang: str):
        model_name = self.langs[lang]["model"]
        self._init_preprocessor(model_name)
        self._init_encoder(model_name)
        self._init_tokenizer(model_name)
        self._trim_memory()

    def _init_model_from_path(self, path: str):
        if not os.path.isdir(path):
            raise ValueError(f"'{path}' is not valid NemoSTT onnx model folder")
        preprocessor_path = f"{path}/preprocessor.ts"
        encoder_path = f"{path}/model.onnx"
        tokenizer_path = f"{path}/tokenizer.spm"
        self._init_preprocessor(preprocessor_path)
        self._init_encoder(encoder_path)
        self._init_tokenizer(tokenizer_path)
        self._trim_memory()

    def _init_preprocessor(self, model_name: str):
        if os.path.isfile(model_name):
            preprocessor_path = model_name
        else:
            preprocessor_path = hf_hub_download(model_name, "preprocessor.ts", subfolder=subfolder_name)
        self.preprocessor = torch.jit.load(preprocessor_path)

    def _init_encoder(self, model_name: str):
        if os.path.isfile(model_name):
            encoder_path = model_name
        else:
            encoder_path = hf_hub_download(model_name, "model.onnx", subfolder=subfolder_name)
        self.encoder = ort.InferenceSession(encoder_path)

    def _init_tokenizer(self, model_name: str):
        if os.path.isfile(model_name):
            tokenizer_path = model_name
        else:
            tokenizer_path = hf_hub_download(model_name, "tokenizer.spm", subfolder=subfolder_name)
        self.tokenizer = spm.SentencePieceProcessor(tokenizer_path)

    def _run_preprocessor(self, audio_16k: np.array):
        input_signal = torch.tensor(audio_16k).unsqueeze(0)
        length = torch.tensor(len(audio_16k)).unsqueeze(0)
        processed_signal, processed_signal_len = self.preprocessor.forward(
            input_signal=input_signal, length=length
        )
        processed_signal = processed_signal.numpy()
        processed_signal_len = processed_signal_len.numpy()
        return processed_signal, processed_signal_len

    def _run_encoder(self, processed_signal: np.array, processed_signal_len: np.array):
        outputs = self.encoder.run(None, {'audio_signal': processed_signal,
                                          'length': processed_signal_len})
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

        self._trim_memory()
        return current_hypotheses

    def stt_file(self, file_path: str):
        audio_buffer, sr = self.read_file(file_path)
        current_hypotheses = self.stt(audio_buffer, sr)
        return current_hypotheses

    def read_file(self, file_path: str):
        audio_file = AudioSegment.from_file(file_path)
        sr = audio_file.frame_rate

        samples = audio_file.get_array_of_samples()
        audio_buffer = np.array(samples)
        return audio_buffer, sr

    @staticmethod
    def _trim_memory():
        """
        If possible, gives memory allocated by PyTorch back to the system
        """
        libc = ctypes.CDLL("libc.so.6")
        libc.malloc_trim(0)
        gc.collect()

    def _resample(self, audio_fp32: np.array, sr: int):
        audio_16k = soxr.resample(audio_fp32, sr, self.sample_rate)
        return audio_16k

    def _to_float32(self, audio_buffer: np.array):
        audio_fp32 = np.divide(audio_buffer, np.iinfo(audio_buffer.dtype).max, dtype=np.float32)
        return audio_fp32


__all__ = ["Model", "available_languages"]
