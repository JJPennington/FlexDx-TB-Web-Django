# Create your views here.

from django.shortcuts import render
from django.conf import settings
from django.http import HttpResponse
from async.forms import ModelForm
import tempfile, os
# For charts
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np
from helper import placement

def home_page(request):
    form = ModelForm()
    return render(request,"index.html", {'form':form})

def about_page(request):
    return render(request,"about.html", {})

def model_page(request):
    if request.method == "GET":
        form = ModelForm(request.GET)
        if form.is_valid():
            strategy = ''
            cd = form.cleaned_data
 
            id, tmp_p = tempfile.mkstemp(suffix='.json',prefix='flexdx',
                        dir=settings.MEDIA_ROOT)
            if cd['type_select'] == '0': #if Single Stategy is clicked
                strategy = cd['int_select']; #run the selected strategy
            else: #'All' is clicked; run all strats.
                strategy = '9';
	    s = "echo " #insert the path to the flexbg.py file
            s += "%f %f %f %f %f %f %f %f %f %f %f %f %f %f %s %s | at now"
            sys_str=s % (cd['t_inc'],cd['t_hiv'],cd['t_mdr'],
                         cd['t_drug1_cost'],
                         cd['t_sm_cost'],cd['t_gxp_cost'],cd['t_cx_cost'],cd['t_mods_cost'],
                         cd['t_dst_cost']-cd['t_cx_cost'],cd['t_sd_cost']-cd['t_sm_cost'],cd['t_sdgxp_cost']-cd['t_gxp_cost'],
                         cd['t_drug2_cost'],cd['t_drug3_cost'],cd['t_outpt_cost'],strategy,tmp_p)

            os.system(sys_str)
            d = {}
            if strategy == '9':
                d['run_all'] = True  
            d['web_path']="{}{}".format(settings.MEDIA_URL,tmp_p.split("/")[-1])
            d['tb_inc']   ='{:,.0f}'.format(cd['t_inc'])
            d['hiv_inc']  ='%.2f' % cd['t_hiv']
            d['mdr_inc']  ='%.1f' % cd['t_mdr']
            d['drg_cst']  ='{:,.2f}'.format(cd['t_drug1_cost'])
            #new costs JP (2013/08/05)
            d['drg2_cst'] ='{:,.2f}'.format(cd['t_drug2_cost'])
            d['drg3_cst'] ='{:,.2f}'.format(cd['t_drug3_cost'])
            d['opt_cst']  ='{:,.2f}'.format(cd['t_outpt_cost'])
            d['sm_cst']   ='{:,.2f}'.format(cd['t_sm_cost'])
            d['sd_cst']   ='{:,.2f}'.format(cd['t_sd_cost'])
            d['gxp_cst']  ='{:,.2f}'.format(cd['t_gxp_cost'])
            d['sdgxp_cst']='{:,.2f}'.format(cd['t_sdgxp_cost'])
            d['cx_cst']   ='{:,.2f}'.format(cd['t_cx_cost'])
            d['dst_cst']  ='{:,.2f}'.format(cd['t_dst_cost'])
            d['mods_cst'] ='{:,.2f}'.format(cd['t_mods_cost'])

            return render(request,"model.html", d)
        
    return render(request,"bad.html", { 'type':'model' })

def dgraph1(request):
    p_val = request.GET
    order = [0,1,7,6,2,8,3,4,5]
    try:
        cmore = [float(p_val['c{}'.format(order[x])]) for x in range(9)]
        inc = [float(p_val['i{}'.format(order[x])]) for x in range(9)]
    except:
        return render(request,"bad.html", { 'type':'dot graph 1' })
                
    fig = Figure(facecolor='white')
    canvas = FigureCanvas(fig)

    ax1 = fig.add_subplot(111)

    ax1.set_ylabel('% decrease in TB incidence at year 5 (more effective -->)')
    ax1.set_xlabel('% increase in cost at year 5 (more costly -->)')

    ax1.plot(cmore,inc,marker='.',linestyle='None')

    ax1.set_title("Percent change (%) in cost and TB incidence at year 5")

    ax1.axhline(color='r')
    ax1.axvline(color='r')

    #Add annotations, simple collision detection
    point_list = []
    for i in xrange(9):
        dx,dy = placement(cmore[i],inc[i],point_list,ax1)
        point_list.append([cmore[i],inc[i],dx,dy])
        ax1.annotate(str(i+1), xy=(cmore[i],inc[i]), 
                     xytext=(cmore[i]+dx,inc[i]+dy),
                     arrowprops=dict(color='red',arrowstyle="->"))

    fig.tight_layout()

    response=HttpResponse(content_type='image/png')
    canvas.print_png(response,facecolor=fig.get_facecolor())
    return response

def dgraph2(request):
    p_val = request.GET
    order = [0,1,7,6,2,8,3,4,5]
    try:
        cmore = [float(p_val['c{}'.format(order[x])]) for x in range(9)]
        mdr = [float(p_val['m{}'.format(order[x])]) for x in range(9)]
    except:
        return render(request,"bad.html",{ 'type':'dot graph 2' })

    fig = Figure(facecolor='white')
    canvas = FigureCanvas(fig)

    ax2 = fig.add_subplot(111)

    ax2.set_ylabel('% decrease in MDR-TB incidence at year 5 (more effective -->)')
    ax2.set_xlabel('% increase in cost at year 5 (more costly -->)')

    ax2.plot(cmore,mdr,marker='.',linestyle='None')

    ax2.set_title("Percent change (%) in cost and MDR incidence at year 5")

    ax2.axhline(color='r')
    ax2.axvline(color='r')

    point_list = []
    for i in xrange(9):
        dx,dy = placement(cmore[i],mdr[i],point_list,ax2)
        point_list.append([cmore[i],mdr[i],dx,dy])
        ax2.annotate(str(i+1), xy=(cmore[i],mdr[i]),
                     xytext=(cmore[i]+dx,mdr[i]+dy),
                     arrowprops=dict(color='red',arrowstyle="->"))    

    fig.tight_layout()

    response=HttpResponse(content_type='image/png')
    canvas.print_png(response,facecolor=fig.get_facecolor())
    return response

def bargraph(request):
    p = request.GET    

    try:
        d = [(float(p['d10']), float(p['d11']), float(p['d12']), float(p['d13']), float(p['d14'])),
             (float(p['d20']), float(p['d21']), float(p['d22']), float(p['d23']), float(p['d24'])),
             (float(p['d30']), float(p['d31']), float(p['d32']), float(p['d33']), float(p['d34'])),
             (float(p['d40']), float(p['d41']), float(p['d42']), float(p['d43']), float(p['d44'])),
             (float(p['d50']), float(p['d51']), float(p['d52']), float(p['d53']), float(p['d54'])),
             (float(p['d60']), float(p['d61']), float(p['d62']), float(p['d63']), float(p['d64'])),
             (float(p['d70']), float(p['d71']), float(p['d72']), float(p['d73']), float(p['d74'])),
             (float(p['d80']), float(p['d81']), float(p['d82']), float(p['d83']), float(p['d84']))]
    except:
        return render(request,"bad.html", { 'type':'bargraph' })

    tickM = ["2. Culture for retreatment",
         "3. GeneXpert for HIV positive only", 
         "4. GeneXpert for smear positive only", 
         "5. GeneXpert for all",
         "6. GeneXpert for all, culture confirmed",
         "7. MODS/TLA",
         "8. Same-day smear microscopy",
         "9. Same-day GeneXpert"]

    colors = ["grey","blue","green","yellow","red"]

    ndata = zip(*d)

    loc = np.arange(len(ndata[0]))
    width = 0.15

    fig = Figure(facecolor='white')
    canvas = FigureCanvas(fig)

    ax = fig.add_subplot(111)

    rect = [ax.bar(loc+width*i, ndata[i], width, color=colors[i]) 
            for i in range(len(ndata))]

    ax.set_ylim(-50,100)
    ax.set_xlim(-width*4, len(loc) +(4*width))

    ax.set_xticks(loc + (2.5*width))

    ax.set_xticklabels(tickM, rotation='30', size='small', stretch='condensed',
                       ha='right' )

    ax.legend ((rect[0][0], rect[1][0], rect[2][0], rect[3][0], rect[4][0]),
                ("TBInc", "MDRInc", "TBMort", "Yr1Cost", "Yr5Cost"),loc='best')

    ax.set_title ("Graph Comparison")
    ax.axhline(color='black')

    ax.set_ylabel('percentage change from baseline')

    fig.tight_layout()

    response=HttpResponse(content_type='image/png')
    canvas.print_png(response,facecolor=fig.get_facecolor())
    return response
