## Privacy-detection curve: gap closed against eps per alert (ceiling = oracle shift of the log-odds by eps)

| arm | eps | n | brier | eps spent (mean) | gap vs gfp | gap vs gfp lo | gap vs gfp hi | ceiling log-odds | ceiling LR | note |
|---|---|---|---|---|---|---|---|---|---|---|
| ours | 0.5 | 0 |  |  |  |  |  |  |  | not_run: no run-log rows match {"system": "ours_compose", "variant": ["eps_0.5"]} |
| ours | inf | 0 |  |  |  |  |  |  |  | not_run: no run-log rows match {"system": "ours_compose", "variant": ["eps_inf"]} |
| eig | 0.5 | 0 |  |  |  |  |  |  |  | not_run: no run-log rows match {"system": "eig", "variant": ["eig_eps_0.5"]} (pending arm: manifest job nonllm.eig_eps_0.5.full_test, nonllm.eig_eps_0.5.subset, nonllm.eig_eps_1.full_test, nonllm.eig_eps_1.subset is not done yet) |
| eig | 1 | 0 |  |  |  |  |  |  |  | not_run: no run-log rows match {"system": "eig", "variant": ["eig_eps_1"]} (pending arm: manifest job nonllm.eig_eps_0.5.full_test, nonllm.eig_eps_0.5.subset, nonllm.eig_eps_1.full_test, nonllm.eig_eps_1.subset is not done yet) |
| eig | 4 | 0 |  |  |  |  |  |  |  | not_run: no run-log rows match {"system": "eig", "variant": ["eig_eps_4"]} (pending arm: manifest job nonllm.eig_eps_0.5.full_test, nonllm.eig_eps_0.5.subset, nonllm.eig_eps_1.full_test, nonllm.eig_eps_1.subset is not done yet) |
| ours | 1 | 771 | 0.1666 | 0.7438 | 0.0037 | -0.0249 | 0.0313 | 1 | 2.7183 |  |
| ours | 2 | 771 | 0.1673 | 1.4569 | -0.0066 | -0.0382 | 0.0263 | 2 | 7.3891 |  |
| ours | 4 | 771 | 0.1679 | 2.6706 | -0.015 | -0.0574 | 0.0296 | 4 | 54.5982 |  |
| eig | 2 | 771 | 0.1823 | 1.4426 | -0.2199 | -0.3918 | -0.1141 | 2 | 7.3891 |  |
| raw_share | inf | 771 | 0.164 | 0 | 0.0402 | -0.1068 | 0.1605 |  |  |  |
| ceiling | 0.5 | 3873 | 0.1125 |  | 0.6294 |  |  | 0.5 | 1.6487 |  |
| ceiling | 1 | 3873 | 0.0784 |  | 1.0959 |  |  | 1 | 2.7183 |  |
| ceiling | 2 | 3873 | 0.0376 |  | 1.6546 |  |  | 2 | 7.3891 |  |
| ceiling | 4 | 3873 | 0.0162 |  | 1.9467 |  |  | 4 | 54.5982 |  |
| reference: one bank |  | 3873 | 0.1585 |  | 0 |  |  |  |  |  |
| reference: gfp_see_everything |  | 3873 | 0.0854 |  | 1 |  |  |  |  |  |
