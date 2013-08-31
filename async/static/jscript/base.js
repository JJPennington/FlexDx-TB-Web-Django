/*Javascript common functions for async_flexdx*/

var status = 0; /*how many results have been loaded: base*/
var cJSON_id; /*Interval ID (for clearing setInterval): base*/
var blink_id; /*blink the running strategy in the progress bar */

$("#title").click(function () { /*Return to index: base*/
    window.location = '/index.html'
});

function isElement ( array, value ) {
    var i;
    for (i = 0; i<array.length; i++) {
        if (value == array[i]) {
            return true;
        }
    }
    return false;
}

function writeBaseline ( data ) { /*Write baseline data: base*/
    $("#left_box").find('h1').text(data["bl"]['name']);
    wh=[2,7,9,12,13];
    for (j=0;j<data["baseline"].length;j++) {
        $(".r_display").append("<li>"+data["baseline"][j]+" "+data["bl"]['lines'][j]+"</li>");
        if (isElement(wh, j)) { /*Adithya's fix*/
            $(".r_display").append("<li class='percent'>Reference</li>");
        }
    }
}

function writeStrat ( data, d_wh, id_strat ) { /*Write strat data: base*/
    /* d_wh is which key in data
       id_strat is which id in the html document to put it
    */
    var val;
    $(id_strat).find('h1').text(data[d_wh]['name']);
    for (var j=0;j<data['strats'].length;j++) {
            
        if ( (val = data['strats'][j]) != '\t' ) {
            $(id_strat).find('ul').append("<li>"+val+" "+data[d_wh]['lines'][j]+"</li>");
        } else {
            $(id_strat).find('ul').append("<li class='percent'>"+data[d_wh]['lines'][j]+"</li>");
        }
    }
}

function loadDone() { /*Remove console animation: base*/
    $("#l_console").fadeOut("slow");
}

function progress (data_stat, data_rt) {
    var status_val, max_val=10;
    status_val = data_stat;
    if (status_val > 10)
        status_val = 10;
    if (data_rt != 9) {
        max_val=2;
    }
    $('#remain').text("Strategies remaining: " + (max_val-status_val).toString());
    $("ul#loading li:lt("+(status_val).toString()+")").each(function () {
        $(this).css('background-color', 'rgb(28,72,130)');
    });
}

function blink () {
    var val = $('ul#loading li:eq(' + status.toString() + ')');
    if ( val.css('background-color') != 'rgb(222, 222, 222)' ) {
        val.css('background-color','rgb(222, 222, 222)');
    } else {
        val.css('background-color','rgb(73, 109, 155)');
    }      
}

function checkJSON() { /*Check JSON file for data: run_modification*/
    $.getJSON(web_path, function(data,stat) {

        if (data["status"] > status) {
            var order = [0,1,7,6,2,8,3,4,5];
            /*If there's new data to display*/
            for (var i=status;i<data["status"];i++) {
                    
                if (i == 0) { 
                    /*Baseline*/

                    writeBaseline( data );
                    $("#left_box").fadeIn();
                    /*Update the progress bar*/
                    progress(data['status'],data['run_type']);

                } else if (i == 10) {
                    /* Init the graphs */
                    initGraphs(data, order);
                    /* Init the comparative chart */
                    initTable(data, order);
                    data["run_type"] = 10; /*Used to indicate all data is loaded to page*/

                } else {

                    if (data["run_type"] == 9) {
                        /*Strategies: ALL*/
                        indiv = "strat" + order[(i-1)].toString();
                        wh = "[name='" + indiv + "']";
                        id_strat = "#" + indiv;
                    
                        writeStrat ( data, indiv, id_strat );
                        /*Update the progress bar*/
                        progress(data['status'],data['run_type']);
                        $(wh).fadeIn();
                    } else {
                        /*Strategies: One*/
                        writeStrat ( data, "strat"+data["run_type"].toString(), "#strat0" );
                        /*Update the progress bar*/
                        progress(data['status'],data['run_type']);
                        data["run_type"] = 10; /*Used to indicate all data is loaded to page*/
                    }
                    if (i == 1) {
                        $("#strat0").fadeIn();
                    }
                }
            } 
            status=data["status"];
        } 
        if (data["run_type"] == 10) {
            /* All data has loaded, stop the loads */
            clearInterval(cJSON_id);
            clearInterval(blink_id);
            $("title").text("FlexDx-TB Model - Results");
            setTimeout( loadDone, 4000 );
        }

    });
}
