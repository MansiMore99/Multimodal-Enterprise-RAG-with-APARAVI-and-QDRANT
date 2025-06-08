<div align="center">

# 🛠️ DIY Home-Repair Helper using APARAVI

> Build your own AI-powered handyman! 🤖🔧

> In this step-by-step project, you’ll learn how to build a complete multimodal Retrieval-Augmented Generation (RAG) pipeline using a no-code visual Data Toolchain. 

[![Access DTC Platform](https://img.shields.io/badge/Launch-Data%20Toolchain-blue?style=for-the-badge&logo=aparatool)](https://dtc-dev.aparavi.com/)
![Multimodal](https://img.shields.io/badge/Multimodal-Supported-orange?style=for-the-badge)

</div>

![Architecture Diagram](https://github.com/MansiMore99/Multimodal-Enterprise-RAG-with-APARAVI-and-QDRANT/blob/main/Images/Img.png)

📥 [Download Aparavi Data Toolchain for AI](https://aparavi.com/download-data-toolchain-for-ai/)

---

#### 📌 Overview

This project develops a DIY Home-Repair Helper, a smart assistant that assists users in resolving household issues using AI. Powered by Aparavi's Data Toolchain for AI, it reads manuals, views images, and answers repair questions in seconds! 💡

Whether you're patching drywall or fixing a leaky faucet, this project brings AI to the rescue for DIY home repairs!
Using Aparavi's powerful Data Toolchain, we’ve built a smart assistant that:

- Reads repair guides
- Analyzes pictures of broken items
- Answers your questions instantly

“It’s like having a brain map for every possible repair — and it works beautifully with the AI assistant behind the scenes.”

---

## 🎯 Project Goals

- 📦 Support text, images, audio, and video
- 🧠 Build a Neo4j-like knowledge graph
- 🔍 Combine vector + graph search
- 💬 Enable multimodal chatbot experience

---

<img width="214" alt="Screenshot 2025-06-07 at 2 37 27 AM" src="https://github.com/user-attachments/assets/34420111-9f8d-4f0b-bd15-8660bfe6c15f" />
<img width="180" alt="Screenshot 2025-06-07 at 11 02 14 PM" src="https://github.com/user-attachments/assets/a648b49b-8fd2-4510-8b50-544f96606c79" />
<img width="161" alt="Screenshot 2025-06-08 at 5 53 20 AM" src="https://github.com/user-attachments/assets/c59cc2d1-7900-4147-aac3-961a8880e7b4" />
<img width="137" alt="Screenshot 2025-06-07 at 2 37 45 AM" src="https://github.com/user-attachments/assets/fd40b218-2aa2-4a94-9911-ea3c6aefd7d1" />
<img width="213" alt="Screenshot 2025-06-07 at 11 02 25 PM" src="https://github.com/user-attachments/assets/e1d81869-cd42-44c8-8551-1d56de6d6ebe" />


## 🧩 Tech Stack Breakdown

| Component         | Role                               |
|------------------|------------------------------------|
| ☁️ AWS S3         | Cloud file storage                  |
| 🧹 Preprocessor   | Cleans & structures the content     |
| 🧠 Embeddings     | Converts content into vectors       |
| 📦 Qdrant DB      | Smart, fast vector search           |
| 🧠 LLMs           | Answers questions using the context |
| 💬 Chat UI        | Where users ask questions           |

## 🔁 Data Processing Pipeline

```
Data ➡️ Parsing ➡️ Cleaning ➡️ Embedding ➡️ Storing ➡️ Searching ➡️ Answering
```
---

<img width="1392" alt="Multimodel RAG" src="https://github.com/user-attachments/assets/68312a36-2187-4405-9867-981e7966f281" />

---

## 🏗️ Setup in 3 Steps

1. 🪣 [Create an S3 Bucket](https://www.youtube.com/watch?v=9GFC6ZGMj_k&t=47s)
2. 🔑 [Get AWS Access & Secret Keys](https://www.youtube.com/watch?v=lntWTStctIE)
3. ✨ Connect it all in Aparavi and start chatting!

---

## 💡 Example Prompt

```json
{
  "system_prompt": "How to patch drywalls?",
  "user_prompt": "Patch drywall by securing a matching piece, taping seams, applying joint compound in layers, sanding smooth, and painting over."
}
```

---

## Step-by-Step Guide:

[![Watch on YouTube](https://img.shields.io/badge/YouTube-Play-red?logo=youtube)]()


#### 📬 Let’s Connect
Have feedback, questions, or want to contribute? Feel free to reach out or fork the project!
Feel free to reach out and follow me on social media:

<p align="center">
  <a href="https://www.linkedin.com/in/mansimore9/">
    <img src="https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white" alt="LinkedIn" />
  </a>
  <a href="https://github.com/MansiMore99">
    <img src="https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white" alt="GitHub" />
  </a>
  <a href="https://medium.com/@mansi.more943">
    <img src="https://img.shields.io/badge/Medium-000000?style=for-the-badge&logo=medium&logoColor=white" alt="Medium" />
  </a>
  <a href="https://x.com/MansiMore99">
    <img src="https://img.shields.io/badge/X-1DA1F2?style=for-the-badge&logo=twitter&logoColor=white" alt="X (Twitter)" />
  </a>
  <a href="https://www.youtube.com/@tech_girl-m9">
    <img src="https://img.shields.io/badge/YouTube-FF0000?style=for-the-badge&logo=youtube&logoColor=white" alt="YouTube" />
  </a>
</p>
