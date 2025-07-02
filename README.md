# phd-work
The following is code associated with different chapters of my PhD.

<p><strong>Python version:</strong> 3.12.10 </p>

<h3>Chapter 3</h3>
<ul>
    <li>Noise-Robust Federated Learning Via Optimal Transport</li>
    <li><a href="https://github.com/luizmanella/phd-work/tree/main/ch3">https://github.com/luizmanella/phd-work/tree/main/ch3</li>
</ul>

<h3>Chapter 4</h3>
<ul>
    <li>Heterogeneous Federated Reinforcement Learning using Wasserstein Barycenters</li>
    <li><a href="https://github.com/luizmanella/phd-work/tree/main/ch4">https://github.com/luizmanella/phd-work/tree/main/ch4</li>
</ul>

<h3>Chapter 5</h3>
<ul>
    <li>Optimal Transport-Based Domain Alignment as a Pre-Processing Step for Federated Learning</li>
    <li><a href="https://github.com/luizmanella/phd-work/tree/main/ch5">https://github.com/luizmanella/phd-work/tree/main/ch5</li>
    <li><strong>Note:</strong>In the homogeneous directory there is the raw source code, but this doesn't allow flexibility in changing the distribution for the data to the agents. Since OT methods take time to compute, having a fixed pre-processed dataset allows for rapid testing after the fact. I recommend using the cleaned code, which is a one shot run to get results. Also, the source code is highly UNoptimized. The preprocessed dataset is stored as one giant file so reading/writing is incredibly slow.</li>
</ul>
Subdirectories in ch5/heterogeneous are citation-coded based on my dissertation:
<ul>
    <li>
        LHS21: Model-Contrastive Federated Learning (https://arxiv.org/pdf/2103.16257.pdf)</li>
    <li>
        WCSY20: Federated Learning with Matched Averaging (https://openreview.net/forum?id=BkluqlSFDS)
    </li>
    <li>
        LCH+21: No Fear of Heterogeneity: Classifier Calibration for Federated Learning with Non-IID Data (https://proceedings.neurips.cc/paper/2021/file/2f2b265625d76a6704b08093c652fd79-Paper.pdf)
    </li>
</ul>

<h3>Chapter 6</h3>
<ul>
    <li>Topological Data Analysis for Network Resilience Quantification</li>
    <li><a href="https://github.com/luizmanella/phd-work/tree/main/ch6">https://github.com/luizmanella/phd-work/tree/main/ch6</li>
</ul>

<h3>Chapter 7</h3>
<ul>
    <li>On the Impact of the Embedding Process on Network Resilience Quantification</li>
    <li><a href="https://github.com/luizmanella/phd-work/tree/main/ch7">https://github.com/luizmanella/phd-work/tree/main/ch7</li>
</ul>


<h4>NOTE:</h4>
Python had a large update when they moved to 3.11. This caused various packages to break, which for some time included even pytorch. Various packages deployed before 3.11 have since depracated and gone unmaintained.
This causes installs to break and dependencies to fail. The code for chapter 4, 6 and 7 are variants of the original code used to generate the published results. I am working on figuring out how to update everything
so imports don't cause a problem.