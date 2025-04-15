# Fingerprint Recognition and AI – What I Learned
---

I’ve been working on a fingerprint recognition project as part of my ELEC4702 course, and it gave me a real opportunity to apply some of the concepts from the [fast.ai](https://www.fast.ai) course in a practical way.

---

## What the Project Was About

The task was to build a fingerprint identification system that can:

- Enroll a fingerprint and associate it with a name
- Store the features in a database or file
- Match new fingerprints against those enrolled
- Evaluate performance using ROC curves and error rate analysis

I used datasets like `DB1_B` from the FVC2000 benchmark, where each subject had multiple fingerprint images taken under varying conditions.

---

## Tools and Techniques I Used

- **Minutiae extraction** from grayscale fingerprint images using OpenCV
- Feature matching using Euclidean distance and directional filtering
- `tkinter` for a simple **GUI interface** to enroll and identify fingerprints
- ROC curve plotting using `scikit-learn` to visualize model accuracy
- JSON as a lightweight storage format for fingerprint templates

The project allowed me to transition from placeholder techniques (like Harris corners) to a more robust Minutiae Cylinder Code (MCC)-style representation, which I was first introduced to in my lectures from ELEC4630.

---

## What I Learned from fast.ai

From the fast.ai course, I gained valuable insights that helped with this project:

- The importance of **data preprocessing** (resizing, enhancing, normalizing)
- Understanding **model evaluation** through AUC and FPR/FNR curves
- Batch size and GPU optimisation — concepts I tested by training vision models
- How to structure experiments and visualize the impact of parameters

In fact, fast.ai’s philosophy of building real, working tools early on helped shape how I approached fingerprint template design and system evaluation.

---

## Key Results

Here are some key takeaways from running my code:

- Matching accuracy was high when comparing enrolled fingerprints to themselves (as expected), but harder when comparing across different versions of the same finger due to alignment and noise
- Tuning the **threshold** for similarity scoring was critical: too low and nothing matched, too high and false positives crept in
- ROC curves helped identify the **sweet spot**: for a 1% False Negative Rate, the False Positive Rate was around ...% in my system

![Sample ROC Curve](images/roc_example.png)

---


*Thanks for reading — follow my journey as I explore more AI topics and practical implementations!* 😊
