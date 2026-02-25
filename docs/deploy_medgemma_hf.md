# Deploying MedGemma 27B on HuggingFace Dedicated Endpoints

This guide walks through deploying `google/medgemma-27b-text-it` as a
HuggingFace Dedicated Inference Endpoint, which our CDS Agent calls via an
OpenAI-compatible API.

## Why HuggingFace Endpoints?

| Feature | Details |
|---|---|
| **Model** | `google/medgemma-27b-text-it` (HAI-DEF) |
| **Cost** | ~$2.50/hr (1× A100 80 GB on AWS) |
| **Scale-to-zero** | Yes — no charges while idle |
| **API format** | OpenAI-compatible (TGI) — zero code changes |
| **Setup time** | ~10 minutes |

## Prerequisites

1. **HuggingFace account** with a valid payment method.
2. **MedGemma access** — accept the gated-model terms at
   <https://huggingface.co/google/medgemma-27b-text-it>.  You must agree to
   Google's Health AI Developer Foundations (HAI-DEF) license.
3. A **HuggingFace token** with `read` scope (already in `.env` as `HF_TOKEN`).

## Step-by-step Deployment

### 1. Create the endpoint

1. Go to <https://ui.endpoints.huggingface.co/new>.
2. **Model Repository**: `google/medgemma-27b-text-it`
3. **Cloud Provider**: AWS (cheapest) or GCP
4. **Region**: `us-east-1` (AWS) or `us-central1` (GCP)
5. **Instance type**: GPU — **1× NVIDIA A100 80 GB**
   - AWS: ~$2.50/hr
   - GCP: ~$3.60/hr
6. **Container type**: Text Generation Inference (TGI) — this is the default.
7. **Advanced Settings**:
   - **Max Input Length**: `12288` (default 4096 is too small for synthesis prompts)
   - **Max Total Tokens**: `16384`
   - **Quantization**: `none` (bfloat16 fits in 80 GB)
   - **Scale-to-zero**: **Enable** (idle timeout: 15 min recommended)

   > **Note:** The default TGI `MAX_INPUT_TOKENS=4096` will cause 422 errors
   > on longer pipeline prompts (especially synthesis). We found `12288` /
   > `16384` to be sufficient for all 6 pipeline steps.
8. Click **Create Endpoint**.

### 2. Wait for the endpoint to become ready

The first deployment downloads the model weights (~54 GB) and starts the TGI
server.  This typically takes **5–15 minutes**.  The status will change from
`Initializing` → `Running`.

### 3. Configure the CDS Agent

Edit `src/backend/.env`:

```dotenv
MEDGEMMA_API_KEY=hf_YOUR_TOKEN_HERE
MEDGEMMA_BASE_URL=https://YOUR_ENDPOINT_ID.us-east-1.aws.endpoints.huggingface.cloud/v1
MEDGEMMA_MODEL_ID=tgi
```

- **`MEDGEMMA_API_KEY`**: Your HuggingFace token (same as `HF_TOKEN`).
- **`MEDGEMMA_BASE_URL`**: The endpoint URL from the HF dashboard, with `/v1`
  appended.  Example:
  `https://x1y2z3.us-east-1.aws.endpoints.huggingface.cloud/v1`
- **`MEDGEMMA_MODEL_ID`**: Use `tgi` — TGI exposes the model under this name
  by default. Alternatively, you can use the full model name
  `google/medgemma-27b-text-it`.

### 4. Verify the connection

```bash
cd src/backend
python -c "
import asyncio
from app.services.medgemma import MedGemmaService

async def test():
    svc = MedGemmaService()
    r = await svc.generate('What is the differential diagnosis for chest pain?')
    print(r[:200])

asyncio.run(test())
"
```

You should see a clinical response from MedGemma.

### 5. Run validation

```bash
cd src/backend
python -m validation.run_validation --medqa --max-cases 50 --seed 42 --delay 2
```

## Cost Estimation

| Scenario | Hours | Cost |
|---|---|---|
| Validation run (120 cases @ ~1 min/case) | ~2 hrs | ~$5 |
| Development / debugging (4 hrs) | ~4 hrs | ~$10 |
| Demo recording | ~1 hr | ~$2.50 |
| **Total estimated** | **~7 hrs** | **~$17.50** |

With scale-to-zero enabled, the endpoint automatically shuts down after 15 min
of inactivity — no overnight charges.

## Troubleshooting

### Cold start latency
After scaling to zero, the first request takes 5–15 min while the model
reloads.  Send a warm-up request before benchmarking.

### 403 Forbidden
Your HF token may not have access to the gated model.  Verify at
<https://huggingface.co/google/medgemma-27b-text-it> that your account has been
granted access.

### Out of memory
If the endpoint fails to start, ensure you selected the **80 GB** A100, not the
40 GB variant.  MedGemma 27B in bfloat16 requires ~54 GB VRAM.

### "model not found" error
TGI exposes the model as `tgi` by default.  If you get a model-not-found error,
try setting `MEDGEMMA_MODEL_ID=google/medgemma-27b-text-it` or check the
endpoint's `/v1/models` route.

## Deleting the Endpoint

When you're done, delete the endpoint from the HF dashboard to stop all
charges:

1. Go to <https://ui.endpoints.huggingface.co/>
2. Select your endpoint → **Settings** → **Delete**

## Comparison with Alternatives

| Platform | GPU | $/hr | Scale-to-Zero | Code Changes | Setup |
|---|---|---|---|---|---|
| **HF Endpoints** | 1× A100 80 GB | **$2.50** | **Yes** | **None** | **Easy** |
| Vertex AI | a2-ultragpu-1g | $5.78 | No | Medium | Medium |
| AWS EC2 (g5.12xlarge) | 4× A10G 96 GB | $5.67 | No (manual) | High | Hard |
| AWS EC2 (p4de.24xlarge) | 8× A100 80 GB | $27.45 | No (manual) | High | Hard |
