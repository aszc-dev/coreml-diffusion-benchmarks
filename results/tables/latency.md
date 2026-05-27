| Cell | Backend | Median latency (ms) | IQR (ms) | Status |
| --- | --- | --- | --- | --- |
| apple-ane-fp16 | apple_coreml | 399.02 | 0.548447 | ok |
| apple-gpu-fp16 | apple_coreml | 442.776 | 3.02783 | ok |
| ours-ane-fp16 | coreml_diffusion | 409.69 | 0.0895105 | ok |
| ours-gpu-fp16 | coreml_diffusion | 492.321 | 6.99343 | ok |
| ours-ane-w4 | coreml_diffusion | N/A | N/A | failed |
| diffusers-mps-fp16 | diffusers_mps | 511.523 | 0.669937 | ok |
| mlx-gpu-fp16 | mlx | N/A | N/A | failed |
