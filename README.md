# AMSbench: A Comprehensive Benchmark for Evaluating MLLM Capabilities in AMS Circuits

Dataset is available at https://huggingface.co/datasets/wwhhyy/AMSBench

The homepage of the thesis is available at https://amsbench.github.io/

The pdf version of the thesis can be obtained here https://arxiv.org/pdf/2505.24138


# AMSbench: A Comprehensive Benchmark for Evaluating MLLM Capabilities in AMS Circuits

<p align="center">
  <img src="https://amsbench.github.io/static/images/AMSBench_banner.png" alt="AMSbench Banner" width="80%">
</p>
<p align="center">
  <strong>AMSbench is a comprehensive benchmark suite designed to systematically evaluate the capabilities of Multi-modal Large Language Models (MLLMs) across critical challenges in Analog/Mixed-Signal (AMS) circuit perception, analysis, and design.</strong>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2505.24138.pdf"><strong>📄 Paper</strong></a> |
  <a href="https://huggingface.co/datasets/wwhhyy/AMSBench"><strong>🤗 Dataset</strong></a> |
  <a href="https://github.com/Why0912/AMSBench"><strong>💻 Code</strong></a> |
  <a href="https://amsbench.github.io/"><strong>🌐 Project Page</strong></a>
</p>

---

## 📖 Abstract

Analog/Mixed-Signal (AMS) circuits play a critical role in the integrated circuit (IC) industry. However, automating AMS circuit design has remained a longstanding challenge due to its difficulty and complexity. Recent advances in Multi-modal Large Language Models (MLLMs) offer promising potential for supporting AMS circuit analysis and design. However, current research typically evaluates MLLMs on isolated tasks within the domain, lacking a comprehensive benchmark that systematically assesses model capabilities across diverse AMS-related challenges.

To address this gap, we introduce **AMSbench**. Our benchmark comprises approximately **8,000 test questions** spanning multiple difficulty levels and assesses eight prominent models, including both open-source and proprietary solutions like GPT-4o and Gemini-2.5 Pro. Our evaluation highlights significant limitations in current MLLMs, particularly in complex multi-modal reasoning and sophisticated circuit design tasks. These results underscore the necessity of advancing MLLMs’ understanding and effective application of circuit-specific knowledge, thereby narrowing the existing performance gap relative to human expertise and moving toward fully automated AMS circuit design workflows.

---

## 🛠️ Benchmark Construction

The creation of AMSbench involved a meticulous, multi-stage process to ensure its comprehensiveness and quality.

### Data Collection & Curation

To build a robust benchmark, we collected data from various authoritative sources, including academic textbooks, research papers, and industrial datasheets. We used specialized tools like **MinerU** to process PDFs and **AMSnet** to generate netlists from schematics. This foundation was enhanced by combining expert annotations with MLLM-generated outputs to create high-quality "circuit-caption" data pairs.

<p align="center">
  <img src="https://amsbench.github.io/static/images/data_collection.png" alt="Data Collection Pipeline" width="80%">
</p>

### Question Generation & Task Design

AMSbench covers both **Visual and Textual Question Answering (VQA/TQA)**. Questions are carefully tiered into three difficulty levels (**Easy, Medium, Hard**) to simulate knowledge requirements from undergraduate students to professional engineers. This tiered approach ensures a thorough and granular evaluation of a model's capabilities, from basic perception to deep, analytical reasoning.

<p align="center">
  <img src="https://amsbench.github.io/static/images/question_generation.png" alt="Question Generation Examples" width="80%">
</p>

---

## 📊 Benchmark Structure & Data Statistics

AMSbench is structured around three core capabilities: **Perception, Analysis, and Design**. The dataset is carefully balanced to provide a robust evaluation framework.

The benchmark consists of:
* **~6,000** questions for **AMS-Perception**
* **~2,000** questions for **AMS-Analysis**
* **~68** questions for **AMS-Design**

The difficulty levels are defined by component counts for Perception tasks and by the required academic/professional level for Analysis tasks, ensuring a comprehensive assessment of both visual understanding and domain knowledge.

<p align="center">
  <img src="https://amsbench.github.io/static/images/data.png" alt="Data Statistics" width="80%">
</p>

---

## 🔬 Evaluation & Key Findings

We evaluated 8 leading MLLMs, and our findings reveal significant limitations in the current state-of-the-art models, especially in complex reasoning and design tasks.

<p align="center">
  <img src="https://amsbench.github.io/static/images/rada.png" alt="Model Performance Radar Chart" width="50%">
</p>

* **Perception**: While models show promise in recognizing local connectivity, their effectiveness deteriorates when performing comprehensive netlist extraction. Even the best-generated netlists require substantial modifications to match the ground truth.
* **Analysis**: Models show potential but often fail to grasp key performance trade-offs, a critical skill for engineers. Some models arrive at correct answers through flawed reasoning.
* **Design**: Performance is poor on complex circuits. Crucially, **no model could consistently generate syntactically correct testbenches**, likely due to a lack of relevant training data.

<p align="center">
  <img src="https://amsbench.github.io/static/images/tab_perception.png" alt="Perception Task Results" width="60%">
  <br/>
  <img src="https://amsbench.github.io/static/images/tab_design_tb.png" alt="Design & Testbench Task Results" width="60%">
</p>

---

## ✨ Task Examples

Here are some examples from the benchmark, illustrating the diversity of tasks.

| Perception Task                                                                                                            | Partition Task                                                                                                                    |
| -------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------- |
| <img src="https://amsbench.github.io/static/images/perception.png" alt="Perception Task Example" width="100%"/>             | <img src="https://amsbench.github.io/static/images/partition.png" alt="Partition Task Example" width="100%"/>                      |
| *Includes total/type-wise counting, location description, connection judgment, and topology generation.* | *The model must identify all constituent structures in a complex circuit, such as a differential pair and Miller compensation.* |
| **Reasoning Task** | **Function Task** |
| <img src="https://amsbench.github.io/static/images/reasoning_cut.png" alt="Reasoning Task Example" width="100%"/>            | <img src="https://amsbench.github.io/static/images/function_cut.png" alt="Function Task Example" width="100%"/>                    |
| *Based on the circuit diagram, the model must explain why it functions as an operational amplifier.* | *The model must determine the intended function of the given circuit diagram.* |

---

## 🚀 Getting Started

1.  **Download the Dataset**: Access the full dataset from our [Hugging Face Repository](https://huggingface.co/datasets/wwhhyy/AMSBench).
2.  **Explore the Data**: The dataset is organized by task (Perception, Analysis, Design) and difficulty. Each entry contains a circuit image and associated questions/answers.
3.  **Evaluate Your Model**: Use the provided data to test the performance of your own MLLMs. We encourage you to follow our evaluation setup for comparable results.
4.  **Share Your Findings**: We welcome contributions and comparisons. If you have a new model, test it on AMSbench and share your results with the community!

---

## ✍️ Citation

If you use AMSbench in your research, please cite our paper:

```bibtex
@misc{shi2025amsbenchcomprehensivebenchmarkevaluating,
      title={AMSbench: A Comprehensive Benchmark for Evaluating MLLM Capabilities in AMS Circuits}, 
      author={Yichen Shi and Ze Zhang and Hongyang Wang and Zhuofu Tao and Zhongyi Li and Bingyu Chen and Yaxin Wang and Zhiping Yu and Ting-Jung Lin and Lei He},
      year={2025},
      eprint={2505.24138},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={[https://arxiv.org/abs/2505.24138](https://arxiv.org/abs/2505.24138)}, 
}
```




