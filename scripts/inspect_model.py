
from src.models.baseline import build_model

model = build_model(
    num_classes=200,
    pretrained=True,
    freeze_backbone=True,
)

print("MOBILENETV3-SMALL FEATURE BLOCKS")
print("=" * 50)

for index, block in enumerate(model.features):
    total = sum(p.numel() for p in block.parameters())

    print(
        f"Block {index}: "
        f"{total:,} parameters"
    )

print("\nCLASSIFIER")
print(model.classifier)

print("\nFEATURE BLOCK COUNT:", len(model.features))
