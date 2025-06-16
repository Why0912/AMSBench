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

## 💡 Why AMSbench? The Challenge of AMS Circuits

Analog/Mixed-Signal (AMS) circuits are the bedrock of modern electronic systems, but their design process is highly dependent on the experience and intuition of engineers. Unlike natural images or plain text, circuit diagrams present unique challenges for MLLMs:

* **Structured Knowledge**: Circuit diagrams are highly structured, containing components, connection relationships (topology), and hierarchical designs.
* **Symbolic Language**: They use a standardized symbolic language that requires precise identification and understanding.
* **Implicit Functionality**: The function of a circuit depends not only on its individual components but on the complex interactions between them.

While current MLLMs excel in general domains, they struggle to comprehend this specialized graphical language without targeted training. Therefore, a standard benchmark is crucial to measure their capabilities in this specialized field and to guide future research.

---

## 🛠️ Benchmark Construction Process

To ensure the benchmark's quality and comprehensiveness, we adopted a rigorous data construction pipeline:

1.  **Data Collection**: We extensively gathered various circuit diagrams and related information from authoritative academic textbooks, cutting-edge research papers, and industrial product datasheets.
2.  **Data Processing**: We utilized advanced tools like **MinerU** and **AMSnet** to parse PDF documents, extract circuit diagrams, and generate initial netlists.
3.  **Data Annotation and Generation**:
    * **Expert Annotation**: Domain experts were invited to perform detailed manual annotations of the circuit diagrams.
    * **MLLM-Assisted Generation**: We leveraged MLLMs (like GPT-4V) to generate preliminary "circuit-caption" data pairs.
    * **Quality Control**: All MLLM-generated content was rigorously reviewed and corrected by our expert team to ensure data accuracy and professional quality.
4.  **Tiered Question Design**: We stratified questions into three difficulty levels—**Easy, Medium, and Hard**—to comprehensively evaluate model capabilities, from basic cognition to deep reasoning.

<p align="center">
  <img src="https://amsbench.github.io/static/images/data_collection.png" alt="Data Collection Pipeline" width="80%">
</p>

---

## 🎯 Core Capabilities & Tasks

AMSbench is designed around three core capability dimensions, encompassing approximately 8,000 test questions.

### 1. AMS-Perception
* **Objective**: To evaluate the fundamental visual understanding of circuit diagrams by MLLMs.
* **Task Examples**:
    * Identifying and locating specific components (e.g., MOSFETs, resistors, capacitors).
    * Extracting the complete circuit connectivity (netlist extraction).
    * Recognizing basic circuit topologies.
* **Question Count**: ~6,000

### 2. AMS-Analysis
* **Objective**: To evaluate the deep understanding and reasoning capabilities of MLLMs regarding circuit functionality and performance.
* **Task Examples**:
    * Analyzing the core function of a circuit (e.g., is this an amplifier or a comparator?).
    * Understanding how changes in component parameters affect circuit performance.
    * Explaining the performance trade-offs.
* **Question Count**: ~2,000

### 3. AMS-Design
* **Objective**: To evaluate the potential of MLLMs in automated circuit design workflows.
* **Task Examples**:
    * Generating a circuit schematic based on given performance specifications (e.g., gain, bandwidth).
    * Creating a valid simulation testbench for a given circuit.
    * Fixing or optimizing design flaws in an existing circuit.
* **Question Count**: 68

<p align="center">
  <img src="https://amsbench.github.io/static/images/data.png" alt="Data Statistics" width="80%">
</p>

---

## 🔬 Model Evaluation & Key Findings

We conducted a comprehensive evaluation of 8 leading MLLMs. The main findings reveal the current state and future challenges for MLLMs in the AMS domain.

<p align="center">
  <img src="https://amsbench.github.io/static/images/rada.png" alt="Model Performance Radar Chart" width="50%">
</p>

* **Widespread Limitations**: All existing models exhibit certain limitations when handling complex AMS circuit tasks.
* **Adequate Perception Capability**: Most models can perform basic component identification tasks reasonably well, but struggle with extracting complete and accurate netlists.
* **Deficiencies in Analysis and Design**: Performance drops significantly in analysis and design tasks that require deep reasoning. Models find it difficult to fully comprehend performance trade-offs and are unable to generate valid testbenches.
* **A Clear Path Forward**: The evaluation results clearly identify the shortcomings of current MLLMs, providing a distinct direction for future model optimization and algorithm research.

<p align="center">
  <img src="https://amsbench.github.io/static/images/tab_perception.png" alt="Perception Task Results" width="80%">
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
