# Submission & Strategy Guide

## Timeline at a Glance

```
Jan 13 ─────────────────────── Feb 24 ──────────── Mar 17-24
 START                     DEADLINE 11:59 PM UTC    RESULTS
 ◄────── Build & Iterate ──────►
```

**⏰ Days remaining as of Feb 15, 2026: ~9 days**

---

## Winning Strategy by Track

### Main Track ($75K)
Focus on **Execution & Communication (30%)** — this is the highest-weighted criterion. A polished video, clean write-up, and well-organized code can make the difference.

**Priority order:**
1. **Execution & Communication (30%)** — Polish everything
2. **Effective Use of HAI-DEF (20%)** — Show the models are essential, not bolted on
3. **Product Feasibility (20%)** — Prove it can work in production
4. **Problem Domain (15%)** — Tell a compelling story about who benefits
5. **Impact Potential (15%)** — Quantify the impact with clear estimates

### Agentic Workflow Prize ($10K)
- Deploy HAI-DEF models as **intelligent agents** or **callable tools**
- Demonstrate a **significant overhaul** of a challenging process
- Show improved efficiency and outcomes via agentic AI

### Novel Task Prize ($10K)
- **Fine-tune** a HAI-DEF model for a task it wasn't originally designed for
- The more creative and useful the adaptation, the better
- Document fine-tuning methodology thoroughly

### Edge AI Prize ($5K)
- Run a HAI-DEF model on **local/edge hardware** (phone, scanner, etc.)
- Focus on model optimization: quantization, distillation, pruning
- Demonstrate real-world field deployment scenarios

---

## Submission Checklist

### Required Deliverables
- [ ] **Kaggle Writeup** — 3 pages or less, following the template
- [ ] **Video demo** — 3 minutes or less
- [ ] **Public code repository** — linked in writeup
- [ ] Uses **at least one HAI-DEF model** (e.g., MedGemma)
- [ ] Code is **reproducible**

### Bonus Deliverables
- [ ] Public interactive live demo app
- [ ] Open-weight Hugging Face model tracing to HAI-DEF

### Write-up Quality
- [ ] Clear project name
- [ ] Team members with specialties and roles listed
- [ ] Problem statement addresses "Problem Domain" and "Impact Potential" criteria
- [ ] Overall solution addresses "Effective Use of HAI-DEF Models" criterion
- [ ] Technical details address "Product Feasibility" criterion
- [ ] All links (video, code, demo) are working and accessible

### Video Quality
- [ ] 3 minutes or less
- [ ] Demonstrates the application in action
- [ ] Explains the problem and solution clearly
- [ ] Shows HAI-DEF model integration
- [ ] Professional quality (clear audio, good visuals)

### Code Quality
- [ ] Well-organized repository structure
- [ ] Clear README with setup instructions
- [ ] Code is commented and readable
- [ ] Dependencies are documented (requirements.txt / environment.yml)
- [ ] Results are reproducible from the repository

---

## Video Tips (30% of score rides on execution)

1. **Open with the problem** (30 sec) — Who suffers? What's broken?
2. **Show the solution** (90 sec) — Live demo, not just slides
3. **Explain the tech** (30 sec) — Which HAI-DEF model, how it's used
4. **Quantify impact** (15 sec) — Numbers, estimates, or projections
5. **Close strong** (15 sec) — Vision for the future

---

## Technical Approach Suggestions

### Application Ideas Aligned to Criteria

| Idea | Models | Special Award Fit |
|------|--------|-------------------|
| Clinical note summarizer with agent routing | MedGemma | Agentic Workflow |
| Radiology triage assistant | MedGemma (vision) | Main Track |
| Dermatology screening on mobile | MedGemma (quantized) | Edge AI |
| Pathology slide analysis for rare diseases | MedGemma (fine-tuned) | Novel Task |
| Patient education chatbot | MedGemma | Main Track |
| Lab result interpreter agent pipeline | MedGemma + tools | Agentic Workflow |
| Wound assessment via phone camera | MedGemma (vision, edge) | Edge AI |

### Key Technical Considerations

1. **Model Selection** — Choose the right HAI-DEF model variant for your task
2. **Fine-tuning** — Document methodology, hyperparameters, dataset curation
3. **Evaluation** — Include performance metrics and analysis
4. **Deployment** — Describe your app stack and how it would scale
5. **Privacy** — Healthcare data is sensitive; address HIPAA/privacy considerations
6. **External Data** — Must be publicly available and equally accessible to all participants

---

## External Data & Tools Rules

- External data is allowed but must be **publicly available at no cost** to all participants
- Use of HAI-DEF/MedGemma is subject to [HAI-DEF Terms of Use](https://developers.google.com/health-ai-developer-foundations/terms)
- Open source code must use an **OSI-approved license**
- AutoML tools are permitted if properly licensed
- **No private code sharing** outside your team during the competition
- Public code sharing must be done on Kaggle forums/notebooks

---

## Draft Writeup Workspace

Use `docs/writeup_draft.md` to iterate on your writeup before submitting on Kaggle:

```markdown
### Project name
[TODO]

### Your team
[TODO: Name, specialty, role for each member]

### Problem statement
[TODO: Define the problem, who's affected, magnitude, why AI is the right solution]
[TODO: Articulate impact — what changes if this works? How did you estimate impact?]

### Overall solution
[TODO: Which HAI-DEF model(s)? Why are they the right choice?]
[TODO: How does the application use them to their fullest potential?]

### Technical details
[TODO: Architecture diagram / description]
[TODO: Fine-tuning details (if applicable)]
[TODO: Performance metrics and analysis]
[TODO: Deployment stack and challenges]
[TODO: How this works in practice, not just benchmarks]
```
