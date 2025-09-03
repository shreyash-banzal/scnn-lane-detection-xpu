a = torch.rand(10, device=device)
b = torch.rand(10, device=device)
print((a + b).device)
