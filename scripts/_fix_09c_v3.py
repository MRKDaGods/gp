"""Fix 09c notebook v3: Unfreeze teacher backbone with AMP + gradient checkpointing.

The v2 teacher had a frozen backbone, yielding only 5% mAP with raw CLIP features.
Fix: Fine-tune the entire ViT-L with mixed precision to fit in T4 16GB.
"""
import json
from pathlib import Path

NB_PATH = Path(__file__).parent.parent / "notebooks/kaggle/09c_kd_vitl_teacher/09c_kd_vitl_teacher.ipynb"

def _src(code: str) -> list[str]:
    """Split code into notebook source lines (each ending with \\n except last)."""
    lines = code.split("\n")
    return [line + "\n" for line in lines[:-1]] + [lines[-1]]


def main():
    nb = json.load(open(NB_PATH, encoding="utf-8"))

    # ── Cell 19: Teacher model ─ unfreeze backbone, add grad checkpointing ──
    nb["cells"][19]["source"] = _src("""\
import timm
from torch.cuda.amp import autocast, GradScaler

TEACHER_MODEL = "vit_large_patch14_clip_224.openai"
TEACHER_EMB_DIM = 1024

class TeacherReID(nn.Module):
    \"\"\"ViT-L/14-CLIP backbone with BNNeck + ID classifier, FULLY fine-tuned.\"\"\"
    def __init__(self, num_classes, emb_dim=TEACHER_EMB_DIM):
        super().__init__()
        self.vit = timm.create_model(TEACHER_MODEL, pretrained=True, num_classes=0)
        self.emb_dim = emb_dim
        # BNNeck for the teacher (same recipe as TransReID)
        self.bnneck = nn.BatchNorm1d(emb_dim)
        nn.init.constant_(self.bnneck.weight, 1.0)
        nn.init.constant_(self.bnneck.bias, 0.0)
        self.bnneck.bias.requires_grad_(False)
        self.classifier = nn.Linear(emb_dim, num_classes, bias=False)

    def forward(self, x):
        \"\"\"Returns (cls_bn, logits, cls_raw). cls_bn for eval, cls_raw for KD feature align.\"\"\"
        cls_raw = self.vit(x)                    # [B, 1024]
        cls_bn  = self.bnneck(cls_raw)           # [B, 1024]
        logits  = self.classifier(cls_bn)        # [B, num_classes]
        return cls_bn, logits, cls_raw


teacher = TeacherReID(num_classes).to(DEVICE)
# Enable gradient checkpointing to save VRAM (~50% activation memory reduction)
teacher.vit.set_grad_checkpointing(enable=True)
# All parameters trainable (backbone + head)
n_params_total    = sum(p.numel() for p in teacher.parameters()) / 1e6
n_params_trainable = sum(p.numel() for p in teacher.parameters() if p.requires_grad) / 1e6
print(f"Teacher params: {n_params_total:.1f}M total, {n_params_trainable:.1f}M trainable (full fine-tune)")
print(f"Gradient checkpointing: enabled")""")

    # ── Cell 20: Update markdown title ──
    nb["cells"][20]["source"] = _src(
        "## 10. Stage 1 \u2014 Teacher Full Fine-Tuning (15 epochs, AMP fp16)")

    # ── Cell 21: Teacher training with AMP + discriminative LR ──
    nb["cells"][21]["source"] = _src("""\
# Full fine-tuning with AMP (fp16) to fit ViT-L in T4 16GB.
# Discriminative LR: backbone gets 10x lower LR than head.
TEACHER_EPOCHS = 15
BACKBONE_LR_T  = 1e-5   # careful fine-tuning of pretrained backbone
HEAD_LR_T      = 1e-3   # fast adaptation of BNNeck + classifier

teacher_ce  = CrossEntropyLabelSmooth(num_classes, 0.05).to(DEVICE)
teacher_tri = TripletLossHardMining(margin=0.3).to(DEVICE)

backbone_params_t = list(teacher.vit.parameters())
head_params_t     = list(teacher.bnneck.parameters()) + list(teacher.classifier.parameters())
t_opt = torch.optim.AdamW([
    {"params": backbone_params_t, "lr": BACKBONE_LR_T},
    {"params": head_params_t,     "lr": HEAD_LR_T},
], weight_decay=5e-4)
t_sched = torch.optim.lr_scheduler.CosineAnnealingLR(t_opt, T_max=TEACHER_EPOCHS, eta_min=1e-6)

scaler_t = GradScaler()
teacher.train()
t_history = {"loss": [], "mAP": [], "rank1": []}
best_t_path = Path("/tmp/teacher_best.pth")
best_t_mAP = 0.0

print(f"Stage 1: full fine-tuning teacher for {TEACHER_EPOCHS} epochs (AMP fp16)")
print(f"  backbone_lr={BACKBONE_LR_T}  head_lr={HEAD_LR_T}")
for epoch in range(1, TEACHER_EPOCHS + 1):
    teacher.train()
    epoch_loss = 0.0
    for imgs_s, imgs_t, labels in train_loader:
        imgs_t = imgs_t.to(DEVICE, non_blocking=True)
        labels_gpu = labels.to(DEVICE, non_blocking=True)

        t_opt.zero_grad()
        with autocast(device_type="cuda"):
            cls_bn, logits, cls_raw = teacher(imgs_t)
            l_ce  = teacher_ce(logits, labels_gpu)
            l_tri = teacher_tri(F.normalize(cls_bn, dim=1), labels_gpu)
            loss  = l_ce + l_tri

        scaler_t.scale(loss).backward()
        scaler_t.unscale_(t_opt)
        torch.nn.utils.clip_grad_norm_(teacher.parameters(), 1.0)
        scaler_t.step(t_opt)
        scaler_t.update()
        epoch_loss += loss.item()

    t_sched.step()
    avg_loss = epoch_loss / len(train_loader)
    t_history["loss"].append(avg_loss)

    if epoch % 5 == 0 or epoch == TEACHER_EPOCHS:
        mAP, rank1 = evaluate(teacher, query_loader_teacher, gallery_loader_teacher, DEVICE)
        t_history["mAP"].append(mAP)
        t_history["rank1"].append(rank1)
        marker = " \\u2605" if mAP > best_t_mAP else ""
        if mAP > best_t_mAP:
            best_t_mAP = mAP
            torch.save(teacher.state_dict(), str(best_t_path))
        print(f"  [T Epoch {epoch:2d}] loss={avg_loss:.4f}  mAP={mAP:.4f}  R1={rank1:.4f}{marker}")
    else:
        print(f"  [T Epoch {epoch:2d}] loss={avg_loss:.4f}")

# Restore best checkpoint and freeze for KD stage
teacher.load_state_dict(torch.load(str(best_t_path), map_location=DEVICE, weights_only=False))
for p in teacher.parameters():
    p.requires_grad_(False)
teacher.eval()
teacher.vit.set_grad_checkpointing(enable=False)  # no longer needed for inference
print(f"\\nTeacher done. Best mAP={best_t_mAP:.4f}  Final: mAP={t_history['mAP'][-1]:.4f}  R1={t_history['rank1'][-1]:.4f}")""")

    # ── Cell 25: Student KD training — add AMP ──
    nb["cells"][25]["source"] = _src("""\
# Hyperparameters
KD_EPOCHS    = 40
BACKBONE_LR  = 1e-5    # low LR \u2014 fine-tuning already-trained backbone
HEAD_LR      = 1e-4    # higher LR for classification head + proj
WARMUP_EP    = 5
KD_ALPHA     = 0.5     # weight for KD logit loss
KD_BETA      = 0.5     # weight for feature alignment loss
KD_TEMP      = 4.0     # distillation temperature

# Losses
ce_loss  = CrossEntropyLabelSmooth(num_classes, 0.05).to(DEVICE)
tri_loss = TripletLossHardMining(margin=0.3).to(DEVICE)
kd_loss  = KDLoss(temperature=KD_TEMP, alpha=KD_ALPHA, beta=KD_BETA).to(DEVICE)

# Optimizer: separate LR groups
backbone_params = list(student.vit.parameters())
head_params     = (list(student.bnneck.parameters()) +
                   list(student.classifier.parameters()) +
                   list(student.feat_proj.parameters()))
optimizer = torch.optim.AdamW([
    {"params": backbone_params, "lr": BACKBONE_LR},
    {"params": head_params,     "lr": HEAD_LR},
], weight_decay=5e-4)

def warmup_lr(epoch):
    if epoch < WARMUP_EP:
        return (epoch + 1) / WARMUP_EP
    progress = (epoch - WARMUP_EP) / (KD_EPOCHS - WARMUP_EP)
    return 0.5 * (1 + np.cos(np.pi * progress))

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, warmup_lr)

scaler_s = GradScaler()
history    = {"loss": [], "loss_task": [], "loss_kd": [], "mAP": [], "rank1": []}
best_mAP   = 0.0
best_state_path = Path("/tmp/student_kd_best.pth")

# Enable gradient checkpointing for student too
student.vit.set_grad_checkpointing(enable=True)

print(f"Stage 2: KD distillation for {KD_EPOCHS} epochs (AMP fp16)")
print(f"  alpha={KD_ALPHA} (logit KD)  beta={KD_BETA} (feat align)  T={KD_TEMP}")

ACCUM_STEPS = 2  # gradient accumulation to compensate for smaller batch

for epoch in range(1, KD_EPOCHS + 1):
    student.train()
    teacher.eval()

    ep_loss = ep_task = ep_kd = 0.0
    optimizer.zero_grad()
    for step_i, (imgs_s, imgs_t, labels) in enumerate(train_loader):
        imgs_s = imgs_s.to(DEVICE, non_blocking=True)
        imgs_t = imgs_t.to(DEVICE, non_blocking=True)
        labels_gpu = labels.to(DEVICE, non_blocking=True)

        # Teacher forward (no grad, fp16)
        with torch.no_grad():
            with autocast(device_type="cuda"):
                _, t_logits, t_cls_raw = teacher(imgs_t)

        # Student forward (AMP)
        with autocast(device_type="cuda"):
            s_cls_raw, s_cls_bn, s_logits, s_feat_proj = student(imgs_s)

            # Task loss (CE + triplet on student)
            l_ce  = ce_loss(s_logits, labels_gpu)
            l_tri = tri_loss(F.normalize(s_cls_raw, dim=1), labels_gpu)
            l_task = l_ce + l_tri

            # KD loss (logit KD + feature alignment)
            l_kd = kd_loss(s_logits, t_logits.float(), s_feat_proj, t_cls_raw.float())

            loss = ((1 - KD_ALPHA) * l_task + l_kd) / ACCUM_STEPS

        scaler_s.scale(loss).backward()
        if (step_i + 1) % ACCUM_STEPS == 0 or (step_i + 1) == len(train_loader):
            scaler_s.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            scaler_s.step(optimizer)
            scaler_s.update()
            optimizer.zero_grad()

        ep_loss += loss.item() * ACCUM_STEPS
        ep_task += l_task.item()
        ep_kd   += l_kd.item()

    scheduler.step()
    n = len(train_loader)
    history["loss"].append(ep_loss / n)
    history["loss_task"].append(ep_task / n)
    history["loss_kd"].append(ep_kd / n)

    if epoch % 5 == 0 or epoch == KD_EPOCHS:
        mAP, rank1 = evaluate(student, query_loader, gallery_loader, DEVICE, is_kd_model=True)
        history["mAP"].append(mAP)
        history["rank1"].append(rank1)
        if mAP > best_mAP:
            best_mAP = mAP
            torch.save(student.state_dict(), str(best_state_path))
            print(f"  [Epoch {epoch:2d}] loss={ep_loss/n:.4f}  task={ep_task/n:.4f}  kd={ep_kd/n:.4f}  mAP={mAP:.4f} R1={rank1:.4f} \\u2605")
        else:
            print(f"  [Epoch {epoch:2d}] loss={ep_loss/n:.4f}  task={ep_task/n:.4f}  kd={ep_kd/n:.4f}  mAP={mAP:.4f} R1={rank1:.4f}")
    else:
        print(f"  [Epoch {epoch:2d}] loss={ep_loss/n:.4f}  task={ep_task/n:.4f}  kd={ep_kd/n:.4f}")

print(f"\\nBest student mAP: {best_mAP:.4f}")""")

    # Save
    with open(NB_PATH, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=True)
    print(f"Updated: {NB_PATH}")
    print("Changes:")
    print("  Cell 19: Teacher model — removed freeze_backbone, added grad checkpointing")
    print("  Cell 20: Updated section title")
    print("  Cell 21: Teacher training — full fine-tune with AMP fp16, discriminative LR")
    print("  Cell 25: Student KD training — added AMP fp16, grad checkpointing")


if __name__ == "__main__":
    main()
