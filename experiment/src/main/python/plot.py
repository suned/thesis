font_size = 10

import matplotlib as mpl

mpl.use("pgf")
pgf_with_rc_fonts = {
    "font.size": font_size,
    "pgf.rcfonts": False,
    "pgf.preamble": [
        "\\usepackage{eulervm}",
        "\\usepackage{mathpazo}"
    ]
}
mpl.rcParams.update(pgf_with_rc_fonts)
import matplotlib.pyplot as plt

plt.figure(figsize=(5,3))
plt.plot(range(5), label="$\\Delta(a)$")
plt.xlabel(u"µ is not $\\mu$")
plt.ylabel("this is the y label")
plt.gca().spines["top"].set_visible(False)
plt.gca().spines["right"].set_visible(False)
plt.gca().xaxis.set_ticks_position("bottom")
plt.gca().yaxis.set_ticks_position("left")
labels = plt.gca().yaxis.get_ticklabels()
labels[0] = ""
plt.gca().set_yticklabels(labels)
plt.legend(loc='lower right', frameon=False, prop={'size': font_size})
plt.tight_layout()
plt.savefig("img/test.pgf")
