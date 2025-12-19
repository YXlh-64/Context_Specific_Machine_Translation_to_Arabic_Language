Using device: cuda:0
Training on both GPUs (distributed via auto device_map)

================================================================================
PRE-TRAINING GPU STATUS:
================================================================================
0, NVIDIA GeForce RTX 5090, 3722 MiB, 32607 MiB
1, NVIDIA GeForce RTX 5090, 3247 MiB, 32607 MiB

Starting training...


Epoch 1/3
================================================================================
Training on 82108 preference pairs (English + French)
0, NVIDIA GeForce RTX 5090, 3722 MiB, 32607 MiB
1, NVIDIA GeForce RTX 5090, 3247 MiB, 32607 MiB

Starting training...


Epoch 1/3
================================================================================
Training on 82108 preference pairs (English + French)
Training:   0%|          | 1/5132 [00:00<1:24:42,  1.01it/s, loss=0.8789, acc=0.2500, lr=0.00e+00]
[DEBUG] Batch 0 - First 5 samples:
  Chosen rewards: [ 0.35546875 -0.25976562  0.8671875   0.96484375  0.28515625]
  Rejected rewards: [ 0.8125      0.78515625  0.17578125  0.7578125  -0.18847656]
  Difference (should be positive): [-0.45703125 -1.046875    0.69140625  0.20703125  0.47265625]
Training: 100%|██████████| 5132/5132 [1:20:23<00:00,  1.06it/s, loss=0.6969, acc=0.4553, lr=1.26e-05]
Validation: 100%|██████████| 571/571 [06:37<00:00,  1.44it/s]

Train Loss: 0.6969 | Train Acc: 0.4553 (EN + FR)
Val Loss: 0.6910 | Val Acc: 0.0025 (EN + FR)

⏱️  Time - Epoch: 1:27:00, Total: 1:27:00, Remaining: 2:54:01

✓ New best validation accuracy: 0.0025
Saving model to models/reward_model_coldstart...

Epoch 2/3
================================================================================
Training on 82108 preference pairs (English + French)

Epoch 2/3
================================================================================
Training on 82108 preference pairs (English + French)
Training:   0%|          | 1/5132 [00:00<1:19:22,  1.08it/s, loss=0.6914, acc=0.5000, lr=1.26e-05]
[DEBUG] Batch 0 - First 5 samples:
  Chosen rewards: [-0.01293945 -0.04980469 -0.07470703 -0.05053711 -0.07470703]
  Rejected rewards: [-0.07470703 -0.07421875 -0.07470703 -0.07470703 -0.08984375]
  Difference (should be positive): [0.06176758 0.02441406 0.         0.02416992 0.01513672]
Training: 100%|██████████| 5132/5132 [1:20:15<00:00,  1.07it/s, loss=0.6892, acc=0.4064, lr=4.53e-06]
Training: 100%|██████████| 5132/5132 [1:20:15<00:00,  1.07it/s, loss=0.6892, acc=0.4064, lr=4.53e-06]
Validation: 100%|██████████| 571/571 [06:37<00:00,  1.44it/s]


Train Loss: 0.6892 | Train Acc: 0.4064 (EN + FR)
Val Loss: 0.6912 | Val Acc: 0.0028 (EN + FR)

⏱️  Time - Epoch: 1:26:53, Total: 2:54:45, Remaining: 1:27:22

✓ New best validation accuracy: 0.0028
Saving model to models/reward_model_coldstart...

Epoch 3/3
================================================================================
Training on 82108 preference pairs (English + French)

Epoch 3/3
================================================================================
Training on 82108 preference pairs (English + French)
Training:   0%|          | 1/5132 [00:00<1:18:46,  1.09it/s, loss=0.6914, acc=0.3125, lr=4.53e-06]
[DEBUG] Batch 0 - First 5 samples:
  Chosen rewards: [-0.0480957  -0.0480957  -0.0480957  -0.02905273 -0.03112793]
  Rejected rewards: [-0.04711914 -0.0279541  -0.02905273 -0.03369141 -0.04711914]
  Difference (should be positive): [-0.00097656 -0.0201416  -0.01904297  0.00463867  0.01599121]
Training:  13%|█▎        | 667/5132 [10:29<1:10:04,  1.06it/s, loss=0.6881, acc=0.3858, lr=3.53e-06]




I found the problem. The validate() function never updates model.train() back to evaluation mode properly during validation on multi-GPU setups. More critically, the rewards are extremely small (near 0), which means the model isn't learning meaningful scales.

Root causes:

Reward head placement issue: The reward head is on cuda:0, but during validation the base model outputs are distributed across both GPUs, causing device mismatch
Reward scaling: Rewards should be larger in magnitude to drive learning, not tiny values like -0.048