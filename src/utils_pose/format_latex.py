import re

# Read the input file
input_file = "/dtu/blackhole/14/189044/marscho/VLM_controller_for_SD/results_onur/analyze.txt"
output_file = "/dtu/blackhole/14/189044/marscho/VLM_controller_for_SD/results_onur/tables.tex"


def extract_tables(text):
    tables = []
    current_metric = None
    current_table = []

    for line in text.split("\n"):
        metric_match = re.match(r"Metric: (.+)", line)
        if metric_match:
            if current_table:
                tables.append((current_metric, current_table))
                current_table = []
            current_metric = metric_match.group(1)
        elif "|" in line and "-" not in line:
            current_table.append(line)

    if current_table:
        tables.append((current_metric, current_table))

    return tables


def format_latex_table(metric, table_lines):
    header = table_lines[0].split("|")[1:-1]  # Extract column names
    # Replace underscores with spaces for meta_transformation_type, escape other underscores
    header = [col.strip().replace("meta_transformation_type", "meta transformation type").replace("_", "\\_") for col in header]

    # Replace underscores with spaces for meta_transformation_type in metric name, escape other underscores
    metric_escaped = metric.replace("meta_transformation_type", "meta transformation type").replace("_", "\\_")

    latex_code = f"\n\\subsection*{{Metric: {metric_escaped}}}\n"
    latex_code += "\\begin{center}\n    \\begin{tabular}{" + "c" * len(header) + "}\n"
    latex_code += "        \\toprule\n        " + " & ".join(header) + " \\\\\n        \\midrule\n"

    for row in table_lines[1:]:
        columns = row.split("|")[1:-1]
        # Replace underscores with spaces for meta_transformation_type in values, escape other underscores
        columns = [col.strip().replace("meta_transformation_type", "meta transformation type").replace("_", "\\_") for col in columns]
        latex_code += "        " + " & ".join(columns) + " \\\\\n"

    latex_code += "        \\bottomrule\n    \\end{tabular}\n\\end{center}\n"

    return latex_code


# Read and process the input file
with open(input_file, "r", encoding="utf-8") as file:
    content = file.read()

tables = extract_tables(content)

# Generate LaTeX output
latex_output = """
"""
for metric, table in tables:
    latex_output += format_latex_table(metric, table)

# Save to output file
with open(output_file, "w", encoding="utf-8") as file:
    file.write(latex_output)

print(f"LaTeX tables have been saved to {output_file}")
