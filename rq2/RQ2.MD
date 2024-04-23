# Can a model sufficiently consider the context and presentation of the statistics gathered to report them accurately?
Motivation: The project aims to gather statistics accurately and efficiently from financial reports. In each report, how-
ever, the statistics in question may be presented differently based on the target audience and the company’s situation.
It will be important to understand how these differ in each report. Many of the reports may be missing some statistics
or present them differently, for example, as a rate or over a different time frame. It is important that we understand how
this presentation will impact how we would like to gather the data. To solve our problem and be able to replace manual
data entry, which can have humans interpret the report, any model must be able to consider the surrounding context. It
will be essential to determine whether we are able to do so. Proposed Methodology: A potential solution to this prob-
lem is using the suggested long and short-term memory provided in deep learning approaches implementing Atten-
tionXML [7]. This method attempts to consider the context around and far away from a financial statistic to extract and
label it better. This will significantly affect the model’s effectiveness but could be computationally expensive; making
balancing a challenge for implementing the solution.

# How to run
Run the main.bat script to set up the environment, run FiNER preprocessing, run AttentionXML preprocessing, and begin training. 
To preform validation, run the main script with the flag `--mode eval`.