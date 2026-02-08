# JupyterLab

JupyterLab is an interactive development environment that provides web-based coding, data, and visualization interfaces. Due to its flexibility and interactive features, it's a popular choice for data scientists and machine learning practitioners.

## Why JupyterLab?

- **Interactive Environment:** JupyterLab allows running code in individual cells, facilitating experimentation and iterative development.
- **Data Exploration and Visualization:** It integrates seamlessly with libraries like matplotlib and seaborn for creating visualizations and exploring data.
- **Documentation and Sharing:** JupyterLab supports markdown and LaTeX for creating rich documentation and sharing code with others.

JupyterLab can be easily installed using conda, if it isn't already installed:

```bash
MuhammadMughees@htb[/htb]$ conda install -y jupyter jupyterlab notebook ipykernel 
```

Make sure you are running the command from within your ai environment.

To start JupyterLab, simply run:

```bash
MuhammadMughees@htb[/htb]$ jupyter lab
```

This will open a new tab in your web browser with the JupyterLab interface.

## Using JupyterLab

JupyterLab launcher with folders and Python 3 options for Notebook and Console.

JupyterLab's primary component is the notebook, which allows combining code, text, and visualizations in a single document. Notebooks are organized into cells, where each cell can contain either code or markdown text.

- **Code cells:** Execute code in various languages (Python, R, Julia).
- **Markdown cells:** Create formatted text, equations, and images using markdown syntax.
- **Raw cells:** Untyped raw text.

Click the "Python 3" icon under the "Notebook" section in the Launcher interface to create a new notebook. This will open a notebook with a single empty code cell.

JupyterLab interface with a file browser and an open, untitled notebook.

Type your Python code into the code cell and press Shift + Enter to execute it. For example:

```python
print("Hello, JupyterLab!")
```

The output of the code will appear below the cell.

JupyterLab interface with a file browser and an open notebook displaying Python code output: 'Hello, JupyterLab!'.

Jupyter notebooks use a stateful environment, which means that variables, functions, and imports defined in one cell remain available to all later cells. Once you execute a cell, any changes it makes to the environment, such as assigning new variables or redefining functions, persist as long as the kernel is running. This differs from a stateless model, where each code execution is isolated and does not retain information from previous executions.

Being aware of the stateful nature of a notebook is important. For example, if you execute cells out of order, you might observe unexpected results due to previously defined or modified variables. Similarly, re-importing modules or updating variable values affects subsequent cell executions, but not those that were previously run.

Say you have a cell that does this:

```python
x = 1
```

then in a later cell you might have:

```python
print(x)  # This will print '1' because 'x' was defined previously.
```

If you change the first cell to:

```python
x = 2
```

and re-run it before running the print(x) cell, the value of x in the environment becomes 2, so the output will now be different when you run the print cell.

Click the "+" button in the toolbar to add new cells. You can choose between code cells and markdown cells using the Dropdown on the toolbar. Markdown cells allow you to write formatted text and include headings, lists, and links.

JupyterLab integrates with libraries like pandas, matplotlib, and seaborn for data exploration and visualization. Here's an example of loading a dataset with pandas and creating a simple plot:

JupyterLab interface showing Python code for creating a DataFrame and scatter plot, with a displayed plot of random data points.

```python
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Create a sample DataFrame
data = pd.DataFrame({
    "column1": np.random.rand(50),  # 50 random values for column1
    "column2": np.random.rand(50) * 10  # 50 random values (multiplied by 10) for column2
})

# Display the first few rows
print(data.head())

# Create a scatter plot
plt.scatter(data["column1"], data["column2"])
plt.xlabel("Column 1")
plt.ylabel("Column 2")
plt.title("Scatter Plot")
plt.show()
```

This code now generates a sample DataFrame with two columns, column1 and column2, containing random values. The rest of the code remains the same, demonstrating how to display the DataFrame's contents and create a scatter plot using the generated data.

To save your notebook, click the save icon in the toolbar or use the Ctrl + S shortcut. Don't forget to rename your Notebook. You can right-click on the Notebook tab or the Notebook in the file browser.

## Restarting the Kernel

JupyterLab uses a kernel to run your code. The kernel is a separate process responsible for executing code and maintaining the state of your computations. Sometimes, you may need to reset your environment if it becomes cluttered with variables or you encounter unexpected behavior.

Restarting the kernel clears all variables, functions, and imported modules from memory, allowing you to start fresh without shutting down JupyterLab entirely.

To restart the kernel:

1. Open the Kernel menu in the top toolbar.
2. Select Restart Kernel to reset the environment while preserving cell outputs, or Restart Kernel and Clear All Outputs to also remove all previously generated outputs from the notebook.

After restarting, re-run any cells containing variable definitions, imports, or computations to restore the environment. This ensures that the notebook state accurately reflects the code you have most recently executed.

This is just a brief overview of Jupyter to get you up and running for this module. For an in-depth guide, refer to the JupyterLab Documentation.
