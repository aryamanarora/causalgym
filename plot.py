from plotnine import ggplot, aes, geom_line, geom_point
from plotnine.scales import scale_x_log10
import json
import pandas as pd

def benchmark():
    data = json.load(open("logs/benchmark.json", "r"))
    df = pd.DataFrame(data)

    # plot
    plot = (
        ggplot(df, aes(x="factor(parameters)", y="iia", group="dataset"))
        + geom_line(aes(color="dataset"))
        + geom_point(aes(fill="dataset"), color="white", size=3)
        + scale_x_log10()
    )
    plot.save("logs/benchmark.pdf")

benchmark()