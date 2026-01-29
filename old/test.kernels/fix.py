# fix_png.py
with open('gaussian_o1_l2.png', 'rb') as f:
    data = f.read()

# IEND chunk: length (4 bytes) + "IEND" + CRC (4 bytes)
IEND = b'\x00\x00\x00\x00IEND\xae\x42\x60\x82'

with open('gaussian_o1_l2_fixed.png', 'wb') as f:
    f.write(data)
    f.write(IEND)

print("Fixed PNG saved as gaussian_o1_l2_fixed.png")