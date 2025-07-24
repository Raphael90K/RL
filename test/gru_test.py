import torch
import torch.nn as nn

# Zufällige Eingabesequenz (batch_size=2, seq_len=5, input_dim=3)
batch_size = 1
seq_len = 1
input_dim = 3
hidden_dim = 4

# Eingabe: z. B. Sensorsequenz o.ä.
x = torch.randn(batch_size, seq_len, input_dim)
print('input.shape:', x.shape)  # (batch_size, seq_len, input_dim)


# Einfaches GRU-Modell (1 Layer, 1 Richtung)
gru = nn.GRU(input_size=input_dim, hidden_size=hidden_dim, batch_first=True)

# Initialer Hidden-State (optional, sonst = zeros)
h0 = torch.zeros(1, batch_size, hidden_dim)

# Vorwärtsdurchlauf
output, h_n = gru(x, h0)

# Formen anzeigen
print("output.shape:", output.shape)  # (batch_size, seq_len, hidden_dim)
print("h_n.shape:   ", h_n.shape)     # (1, batch_size, hidden_dim)

# Extrahiere letzten Schritt aus output
output_last = output[:, -1, :]        # (batch_size, hidden_dim)
hidden_last = h_n[0]                  # (batch_size, hidden_dim)

# Vergleich
print("\nVergleich:")
print("output[:, -1, :] ≈ hidden[0, :, :] ?", torch.allclose(output_last, hidden_last, atol=1e-6))

# Unterschiede anzeigen (sollte sehr klein oder 0 sein)
print("\nDifferenz:", (output_last - hidden_last).abs().max().item())

print(output)

print(h_n)