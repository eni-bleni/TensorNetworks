module layout

using PyPlot
using PyCall
@pyimport matplotlib.transforms as mpltrafo



### LAYOUT choices:

function nice_ticks()
    ax = subplot(111)
    ax[:get_xaxis]()[:set_tick_params](direction="in", bottom=1, top=1)
    ax[:get_yaxis]()[:set_tick_params](direction="in", left=1, right=1)

    for l in ax[:get_xticklines]()
        l[:set_markersize](8)
        l[:set_markeredgewidth](2.0)
    end
    for l in ax[:get_yticklines]()
        l[:set_markersize](8)
        l[:set_markeredgewidth](2.0)
    end
    for l in ax[:yaxis][:get_minorticklines]()
        l[:set_markersize](4)
        l[:set_markeredgewidth](1.5)
    end
    for l in ax[:xaxis][:get_minorticklines]()
        l[:set_markersize](4)
        l[:set_markeredgewidth](1.5)
    end

    ax[:set_position](mpltrafo.Bbox([[0.16, 0.12], [0.95, 0.94]]))
end

linew = 2
rc("font", size = 18) #fontsize of axis labels (numbers)
rc("axes", labelsize = 20, lw = linew) #fontsize of axis labels (symbols)
rc("lines", mew = 2, lw = linew, markeredgewidth = 2)
rc("patch", ec = "k")
rc("xtick.major", pad = 7)
rc("ytick.major", pad = 7)

PyCall.PyDict(matplotlib["rcParams"])["mathtext.fontset"] = "cm"
PyCall.PyDict(matplotlib["rcParams"])["mathtext.rm"] = "serif"
PyCall.PyDict(matplotlib["rcParams"])["figure.figsize"] = [8.0, 6.0]



end
