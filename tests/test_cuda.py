import torch

def test_cuda_available():
    assert torch.cuda.is_available(), "CUDA ist nicht verfügbar!"
    print(f"CUDA verfügbar: {torch.cuda.is_available()}")
    print(f"Anzahl GPUs: {torch.cuda.device_count()}")
    print(f"Aktuelle Device-ID: {torch.cuda.current_device()}")
    print(f"Device-Name: {torch.cuda.get_device_name(torch.cuda.current_device())}") 