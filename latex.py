import re

def text_to_latex(text):
    """
    Converts plain text (Markdown-style) to LaTeX format,
    with support for fractions, integrals, limits, matrices, inequalities, and logic symbols.
    """

    # ✅ Fix Markdown Headings
    text = re.sub(r'^# (.*)', r'\\section{\1}', text, flags=re.MULTILINE)
    text = re.sub(r'^## (.*)', r'\\subsection{\1}', text, flags=re.MULTILINE)
    text = re.sub(r'^### (.*)', r'\\subsubsection{\1}', text, flags=re.MULTILINE)

    # ✅ Fix Math Symbols (Ensure proper escaping)
    math_symbols = {
        'RR': r'\\mathbb{R}', 'NN': r'\\mathbb{N}', 'ZZ': r'\\mathbb{Z}',
        'QQ': r'\\mathbb{Q}', 'CC': r'\\mathbb{C}', 'subseteq': r'\\subseteq',
        'supseteq': r'\\supseteq', 'cup': r'\\cup', 'cap': r'\\cap',
        'forall': r'\\forall', 'exists': r'\\exists', 'neg': r'\\neg',
        'lor': r'\\lor', 'land': r'\\land', 'Rightarrow': r'\\Rightarrow',
        'Leftrightarrow': r'\\Leftrightarrow', 'leq': r'\\leq', 'geq': r'\\geq',
        'll': r'\\ll', 'gg': r'\\gg', 'ne': r'\\neq', 'approx': r'\\approx',
        'equiv': r'\\equiv'
    }

    for symbol, latex_code in math_symbols.items():
        text = re.sub(r'\b' + symbol + r'\b', latex_code, text)

    # ✅ Fix Matrices
    text = re.sub(r'\$\$matrix:(.*?)\$\$', lambda m: r'\[\begin{bmatrix} ' + m.group(1) + r' \end{bmatrix}\]', text, flags=re.DOTALL)
    #piece wise function
    text = re.sub(r'\$\$piecewise:(.*?)\$\$', lambda m: r'\[\begin{cases} ' + m.group(1).replace("\\\\leq", "\\leq") + r' \end{cases}\]', text, flags=re.DOTALL)




    return text


# ✅ Test Input (Direct in Python)
input_text = r"""
# LaTeX Test File

## Matrices
- A 2×2 matrix:
$$matrix:
1 & 2 \\
3 & 4
$$

## Piecewise Functions
$$piecewise:
f(x) = x^2, & x > 0 \\
f(x) = -x, & x \leq 0
$$

## Math Symbols
The set of all real numbers is RR.
"""

# Convert input to LaTeX
latex_code = text_to_latex(input_text)

# Print the LaTeX output
print("\nGenerated LaTeX Code:\n")
print(latex_code)
