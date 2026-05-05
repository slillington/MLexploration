### Agent Harness in Agentic Workflows

The **harness** is the layer between the LLM and deterministic function execution that:
- Provides guard rails helping the LLM identify the right tool for each task
- Structures tool usage rigidly to ensure consistent input/output contracts
- Enables reliable, repeatable agent behavior

**Example workflow:**
An LLM tasked with analyzing a FASTA file and plotting amino acid diversity:
1. Identifies Biopython as the available tool for MSA generation
2. Calls deterministic Biopython functions to produce the MSA
3. Calls separate deterministic functions to generate visualization from that MSA

This structure allows LLMs to reason about *what* to do while deterministic code handles *how* to do it consistently.