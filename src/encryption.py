import os
import torch
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend


def generate_key():
    return os.urandom(32)


def generate_iv():
    return os.urandom(12)


def tensor_to_bytes(tensor):
    tensor_np = tensor.cpu().detach().numpy().astype("float32")
    return tensor_np.tobytes(), tensor_np.shape


def bytes_to_tensor(byte_data, shape, device="cpu"):
    import numpy as np

    tensor_np = np.frombuffer(byte_data, dtype="float32").copy().reshape(shape)
    return torch.from_numpy(tensor_np).to(device)


def encrypt_data(data_bytes, key, iv):
    encryptor = Cipher(
        algorithms.AES(key), modes.GCM(iv), backend=default_backend()
    ).encryptor()
    return encryptor.update(data_bytes) + encryptor.finalize(), encryptor.tag


def decrypt_data(ciphertext, tag, key, iv):
    decryptor = Cipher(
        algorithms.AES(key), modes.GCM(iv, tag), backend=default_backend()
    ).decryptor()
    return decryptor.update(ciphertext) + decryptor.finalize()


class LatentEncryptor:
    def __init__(self, key=None):
        self.key = key if key else generate_key()

    def encrypt(self, latent_tensor):
        data_bytes, shape = tensor_to_bytes(latent_tensor)
        iv = generate_iv()
        ciphertext, tag = encrypt_data(data_bytes, self.key, iv)
        return ciphertext, {"iv": iv, "tag": tag, "shape": shape}

    def decrypt(self, ciphertext, metadata, device="cpu"):
        plaintext = decrypt_data(ciphertext, metadata["tag"], self.key, metadata["iv"])
        return bytes_to_tensor(plaintext, metadata["shape"], device=device)


def ciphertext_to_bits(ciphertext, max_len=None):
    import numpy as np

    byte_array = np.frombuffer(ciphertext, dtype=np.uint8)
    bit_array = np.unpackbits(byte_array).astype(np.float32)
    if max_len is not None:
        padded = np.zeros(max_len, dtype=np.float32)
        padded[: len(bit_array)] = bit_array
        bit_array = padded
    return torch.from_numpy(bit_array)


def bits_to_ciphertext(bit_tensor, original_byte_len):
    import numpy as np

    bit_array = (bit_tensor.cpu().numpy() >= 0.5).astype(np.uint8)[
        : original_byte_len * 8
    ]
    return np.packbits(bit_array).tobytes()
