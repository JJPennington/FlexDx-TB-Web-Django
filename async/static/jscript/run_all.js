/*Javascript file for run_all option for async_flexdx*/

/*$("[name^='strat']").hide(); Hide the scenario tabs: run_all*/

$("[name^='strat']").click(function () { 
        /*When tabs are clicked display: run_all*/

    var wh = $(this).attr('name');
    switch (wh) {
        case 'strat9':
            $("#left_box").hide();
            $("#table_box").hide();
            $("[id^='strat']").hide();
            $("#graph_box").show();
            break;
        case 'strat10':
            $("#left_box").hide();
            $("#graph_box").hide();
            $("[id^='strat']").hide();
            $("#table_box").show();
            break;
        default:
            wh = "#" + wh;
            $("[id^='strat']").hide();
            $("#graph_box").hide();
            $("#table_box").hide();
            $("#left_box").show();
            $(wh).show();
    }

});

function initGraphs(data, order) {
    /* Init the graph box */
    $("#graph_box").append("<img src='" + data['strat9']['graphs'][2] + "' alt='BarGraph' class='graph' >");
    $("#graph_box").append("<div id='graph1_frame'>");
    $("#graph1_frame").append("<img src='" + data['strat9']['graphs'][0] + "' alt='DotGraph1' class='graph' >");
    // Create the legend for dotgraph 1
    var s_a = ['Cult. for retreatment',
               'Xpert for HIV+',
               'Xpert for smear+',
               'Xpert for all',
               'Xpert for all, cult. conf.',
               'MODS/TLA',
               'Same-day smear',
               'Same-day Xpert'];
    dg1cont="<tr><th colspan='4'>Percent change in cost and TB incidence compared to Baseline (Smear, 1) at year 5</th></tr>";
    dg1cont+='<tr><th>Number</th><th>Name</th><th>Chg in Cost</th><th>Chg in TB Incidence</th></tr>';
    var dg1_a = data['strat9']['graphs'][0].split('?')[1].split('&');
    cost_struct = [];
    inc1_struct = [];
    for (var j=0;j<9;j++) {
        cost_struct.push(dg1_a[j].split('=')[1]);
        inc1_struct.push(dg1_a[j+9].split('=')[1]);
    }
    for (var j=1;j<9;j++) {
        dg1cont+="<tr><th>"+(j+1).toString()+"</th><td>"+s_a[j-1]+
                 "</td><td>"+cost_struct[order[j]]+" %</td><td>"+inc1_struct[order[j]]+" %</td></tr>";
    }
    $("#graph1_frame").append("<table class='leg_disp'>" + dg1cont + "</table>");
    $("#graph_box").append("<div id='graph2_frame'>");

    $("#graph2_frame").append("<img src='" + data['strat9']['graphs'][1] + "' alt='DotGraph2' class='graph' >");
    // Create the legend for dotgraph 2
    dg2cont="<tr><th colspan='4'>Percent change in cost and MDR incidence compared to Baseline (Smear, 1) at year 5</th></tr>";
    dg2cont+='<tr><th>Number</th><th>Name</th><th>Chg in Cost</th><th>Chg in TB Incidence</th></tr>';
    var dg2_a = data['strat9']['graphs'][1].split('?')[1].split('&');
    mdr2_struct = [];
    for (var j=9;j<18;j++) {
        mdr2_struct.push(dg2_a[j].split('=')[1]);
    }
    for (var j=1;j<9;j++) {
        dg2cont+="<tr><th>"+(j+1).toString()+"</th><td>"+s_a[j-1]+
                 "</td><td>"+cost_struct[order[j]]+" %</td><td>"+mdr2_struct[order[j]]+" %</td></tr>";
    }
    $("#graph2_frame").append("<table class='leg_disp'>" + dg2cont + "</table>");

    $("[name='strat9']").fadeIn();
}

function initTable ( data, order ) { /*Write table data: run_all*/
    /* Init the table box */
    var max_g = 0.0;
    var max_r = 0.0;
    var tb = $("#table_box");
    var h_a = ['Chg in total inc.',
               'Chg in MDR inc.',
               'Chg in TB mort.',
               'Cost Chg Yr1',
               'Cost Chg Yr5'];
    var s_a = ['Cult. for retreatment',
               'Xpert for HIV+',
               'Xpert for smear+',
               'Xpert for all',
               'Xpert for all, cult. conf.',
               'MODS/TLA',
               'Same-day smear',
               'Same-day Xpert' ];
    var t_string = ""
    tb.append("<table class='disp'></table>");
    t_string+="<tr><th>Scenario #</th><th>Name";
    for (var j=0;j<5;j++) {
        t_string +="</th><th>"+h_a[j];
    }
    t_string += "</th></tr>"
    for (var j=0;j<8;j++) {
        var wh = [3,9,12,16,18];
        var strat = "strat" + order[(j+1)].toString();
        t_string+="<tr><th>"+(j+2).toString()+"</th><th>"+s_a[j]+"</th>";
        for (var x=0;x<5;x++) {
            var ln = $.trim(data[strat].lines[wh[x]]);
            var data_value = parseFloat(ln.split('%')[0],10);
            if (ln.indexOf("reduction")!=-1||ln.indexOf("decrease")!=-1) {
                if (data_value > max_g) {
                    max_g = data_value;
                }
                t_string+="<td class='green'>" + data_value.toString() + "%</td>";
            } else {
                if (data_value > max_r) {
                    max_r = data_value;
                }
                t_string+="<td class='red'>"+ data_value.toString() +"%</td>";
            }
        }
        t_string+="</tr>";
    }                    
    t_string+="<tr><td colspan='7' class='comment'>";
    t_string+="All values indicate pecentage change from baseline scenario</td></tr>"
    t_string+="<tr><td colspan='7' class='comment'>";
    t_string+="Green values represent reductions, red values represent increases</td></tr>"
     
    tb.find("table").append(t_string);

    tb.find("td").each(function () {
        var start = 96;
        var max = 230;
        var prop = "background-color";
        if ($(this).is('.green')) {
            var v = parseFloat($(this).text().split('%')[0],10);
            $(this).css(prop,"rgb(0,"+(Math.round(start+((max-start)*v/max_g))).toString()+",0)");
        } else if ($(this).is('.red')) {
            var v = parseFloat($(this).text().split('%')[0],10);
            $(this).css(prop,"rgb("+(Math.round(start+((max-start)*v/max_r))).toString()+",0,0)");
        }
    });
                                             
    $("[name='strat10']").fadeIn();
}
