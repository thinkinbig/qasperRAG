import matplotlib.pyplot as plt
import matplotlib.patches as patches


# Define a function to create lines with optional arrows
def create_line(ax, start, end, text='', arrow=True, linestyle='-', dash_length=5, space_length=5, color='black'):
    if arrow:
        # Create the line with an arrow
        line = patches.FancyArrowPatch(start, end, arrowstyle='->', linestyle=linestyle,
                                       mutation_scale=10, color=color)
        ax.add_patch(line)
    else:
        # Create a dashed line without an arrow
        line = plt.Line2D((start[0], end[0]), (start[1], end[1]), linestyle=linestyle,
                          dashes=(dash_length, space_length), color=color)
        ax.add_line(line)

    if text:
        mid_x = (start[0] + end[0]) / 2
        mid_y = (start[1] + end[1]) / 2
        plt.text(mid_x, mid_y, text, ha='center', va='center', fontsize=9)


# Define a function to create a rectangle with text inside
def create_rectangle(ax, center, width, height, text, fillcolor='none'):
    # Create a rectangle and add it to the plot
    rect = patches.Rectangle((center[0] - width / 2, center[1] - height / 2), width, height, linewidth=1,
                             edgecolor='none', facecolor=fillcolor, zorder=1)
    ax.add_patch(rect)
    # Add text centered in the rectangle
    plt.text(center[0], center[1], text, ha='center', va='center', fontsize=9, zorder=2)


# Create a new figure
fig, ax = plt.subplots(figsize=(12, 6))

# Create rectangles with fill colors and transparent edges
create_rectangle(ax, (1, 5), 2, 1, 'Paragraph\nSplitter', fillcolor='lightblue')

# Create document rectangles with fill colors and transparent edges
create_rectangle(ax, (4, 2), 2, 0.5, 'Document\nChunk 1', fillcolor='lightyellow')
create_rectangle(ax, (4, 3), 2, 0.5, 'Document\nChunk 2', fillcolor='lightcyan')
create_rectangle(ax, (4, 4), 2, 0.5, 'Document\nChunk 3', fillcolor='lightcoral')
create_rectangle(ax, (4, 5), 2, 0.5, 'Document\nChunk 4', fillcolor='lightcyan')
create_rectangle(ax, (4, 6), 2, 0.5, 'Document\nChunk 5', fillcolor='lightyellow')
create_rectangle(ax, (4, 7), 2, 0.5, 'Document\nChunk 6', fillcolor='lightcoral')
create_rectangle(ax, (4, 8), 2, 0.5, 'Document\nChunk 7', fillcolor='lightcoral')

create_rectangle(ax, (7, 5), 2, 1, 'Bi-encoder\n(emb_model)', fillcolor='lightgreen')

create_rectangle(ax, (10, 3), 2, 0.5, 'Top-k\nDocument', fillcolor='lightyellow')
create_rectangle(ax, (10, 4), 2, 0.5, 'Top-k\nDocument', fillcolor='lightcoral')
create_rectangle(ax, (10, 5), 2, 0.5, 'Top-k\nDocument', fillcolor='lightcoral')
create_rectangle(ax, (10, 6), 2, 0.5, 'Top-k\nDocument', fillcolor='lightcyan')
create_rectangle(ax, (10, 7), 2, 0.5, 'Top-k\nDocument', fillcolor='lightcyan')

create_rectangle(ax, (13, 5), 2, 1, 'Cross-encoder\n(rerank_model)', fillcolor='moccasin')


create_rectangle(ax, (16, 4), 2, 0.5, 'Top-k\nDocument', fillcolor='lightcyan')
create_rectangle(ax, (16, 5), 2, 0.5, 'Top-k\nDocument', fillcolor='lightcoral')
create_rectangle(ax, (16, 6), 2, 0.5, 'Top-k\nDocument', fillcolor='lightyellow')

create_rectangle(ax, (19, 5), 2, 1, 'Language Model\n(Llama 7b 13k)', fillcolor='lightcoral')

# Define the process flow lines
create_line(ax, (2, 5), (3, 2), arrow=False, linestyle='--')
create_line(ax, (2, 5), (3, 3), arrow=False, linestyle='--')
create_line(ax, (2, 5), (3, 4), arrow=False, linestyle='--')
create_line(ax, (2, 5), (3, 5), arrow=True, linestyle='-')
create_line(ax, (2, 5), (3, 6), arrow=False, linestyle='--')
create_line(ax, (2, 5), (3, 7), arrow=False, linestyle='--')
create_line(ax, (2, 5), (3, 8), arrow=False, linestyle='--')

create_line(ax, (5, 2), (6, 5), arrow=False, linestyle='--')
create_line(ax, (5, 3), (6, 5), arrow=False, linestyle='--')
create_line(ax, (5, 4), (6, 5), arrow=False, linestyle='--')
create_line(ax, (5, 5), (6, 5), arrow=True, linestyle='-')
create_line(ax, (5, 6), (6, 5), arrow=False, linestyle='--')
create_line(ax, (5, 7), (6, 5), arrow=False, linestyle='--')
create_line(ax, (5, 8), (6, 5), arrow=False, linestyle='--')

create_line(ax, (8, 5), (9, 3), arrow=False, linestyle='--')
create_line(ax, (8, 5), (9, 4), arrow=False, linestyle='--')
create_line(ax, (8, 5), (9, 5), arrow=False, linestyle='--')
create_line(ax, (8, 5), (9, 6), arrow=False, linestyle='--')
create_line(ax, (8, 5), (9, 7), arrow=False, linestyle='--')

create_line(ax, (11,3), (12, 5), arrow=False, linestyle='--')
create_line(ax, (11, 4), (12, 5), arrow=False, linestyle='--')
create_line(ax, (11, 5), (12, 5), arrow=True, linestyle='-')
create_line(ax, (11, 6), (12, 5), arrow=False, linestyle='--')
create_line(ax, (11, 7), (12, 5), arrow=False, linestyle='--')


create_line(ax, (14, 5), (15, 4), arrow=False, linestyle='--')
create_line(ax, (14, 5), (15, 5), arrow=False, linestyle='--')
create_line(ax, (14, 5), (15, 6), arrow=False, linestyle='--')

create_line(ax, (17, 4), (18, 5), arrow=False, linestyle='--')
create_line(ax, (17, 5), (18, 5), arrow=True, linestyle='-')
create_line(ax, (17, 6), (18, 5), arrow=False, linestyle='--')


create_line(ax, (8, 5), (9, 5), text='', arrow=True)

# Add an arrow from the question to the Bi-encoder
# create_line(ax, (1, 2), (4, 4.5), text='Question', arrow=True)

# Set limits and turn off axes
ax.set_xlim(0, 20)
ax.set_ylim(0, 10)
plt.axis('off')


plt.savefig("llama_index.png", dpi=300)

# Display the plot with colored filled rectangles and no bottom line
plt.show()
