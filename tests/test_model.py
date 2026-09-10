import torch
from model.fusion import FusionLSTMClassifier
from common.config import NUM_CHANNELS, BAND_NAMES, NUM_BEHAVIOR_FEATS, SEQ_LEN
from common.states import STATES

def _make_model():
    return FusionLSTMClassifier(
        num_channels=NUM_CHANNELS, num_bands=len(BAND_NAMES),
        num_behavior_feats=NUM_BEHAVIOR_FEATS, num_classes=len(STATES),
    )

def test_forward_output_shape():
    model = _make_model()
    eeg_seq = torch.randn(4, SEQ_LEN, NUM_CHANNELS, len(BAND_NAMES))
    behavior_seq = torch.randn(4, SEQ_LEN, NUM_BEHAVIOR_FEATS)
    logits = model(eeg_seq, behavior_seq)
    assert logits.shape == (4, len(STATES))

def test_model_can_overfit_single_batch():
    torch.manual_seed(0)
    model = _make_model()
    eeg_seq = torch.randn(8, SEQ_LEN, NUM_CHANNELS, len(BAND_NAMES))
    behavior_seq = torch.randn(8, SEQ_LEN, NUM_BEHAVIOR_FEATS)
    labels = torch.randint(0, len(STATES), (8,))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = torch.nn.CrossEntropyLoss()
    losses = []
    for _ in range(50):
        optimizer.zero_grad()
        logits = model(eeg_seq, behavior_seq)
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0] * 0.5
