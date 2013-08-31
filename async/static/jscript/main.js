$(document).ready(function () {
   
    $("#left_box").show();
    $("#right_box").show(); 
    $("#linkbar li").show();
    var j_data; /*Global var to hold the JSON data*/

    function stopRKey(evt) {
        /* Code to prevent the enter key from submitting the form */
        var evt = (evt) ? evt : ((event) ? event : null);
        var node = (evt.target) ? evt.target : ((evt.srcElement) ? evt.srcElement : null);
        if ((evt.keyCode == 13) && (node.type=="text"))  {return false;}
    }

    document.onkeypress = stopRKey; 


    $.ajax ({
            /*
              Load the JSON data into a global variable
              Reading the file only once per page load.
            */
            url: '/static/mouseOver.json',
            async: false,
            dataType: 'json',
            success: function (data) {
                j_data = data;
            }
    });

    $("[name^='t_']").focusout(function() {
        /*
          Sanity checking on the text-field inputs
        */
        var x = $(this);
        var t = $.trim(x.val());
        x.val(t);
        if (t == '' || isNaN(t)) {
            alert("Input value must exist and be a number");
            x.val('');
            setTimeout(function () {
                x.focus();
            }, 1);
        } else {
            switch ($(this).attr('name')) {
                case 't_sm_cost':
                    if (parseFloat($("[name='t_sd_cost']").val(),10) < 
                        parseFloat($(this).val(),10)) {
                        $("[name='t_sd_cost']").val($(this).val());
                    }
                    break;
                case 't_gxp_cost':
                    if (parseFloat($("[name='t_sdgxp_cost']").val(),10) <
                        parseFloat($(this).val(),10)) {
                        $("[name='t_sdgxp_cost']").val($(this).val());
                    }
                    break;
                case 't_cx_cost':
                    if (parseFloat($("[name='t_dst_cost']").val(),10) <
                        parseFloat($(this).val(),10)) {
                        $("[name='t_dst_cost']").val($(this).val());
                    }
                    
            }
        }
    });

    function loadSelected () {
        /*
          Display the checked strategy in the mouse over <div>
        */
        var single_tmp = $("input[name='int_select']:checked").val();
        var a_s_tmp = $("input[name='type_select']:checked").val();
        if (a_s_tmp == '0') { //Single is clicked
            var wh = 'opt' + single_tmp;
            $("#m_over").html(j_data[wh]);
        } else { // All is clicked
            $("#m_over").html(j_data['opt9'])
        }
    }

    $("[id^='opt']").mouseenter(function() {
        /*
          Display the strategy under the mouse in the mouse over <div>
        */
        var wh = $(this).attr('id');
        $("#m_over").html(j_data[wh]);
    });

    $("[id^='opt']").mouseleave(function () {
        /*
          If the user is no longer hovering over a strategy, default to selected
        */
        loadSelected();
    });

    $("[class='single']").click(function () {
        $("#inputs").show();
        loadSelected();
    });

    $("[class='all']").click(function () {
        $("#inputs").hide();
        loadSelected();
    });

    $("[name='int_select']").click(function () {
        /*
          If the user has clicked a strategy, display it
        */
        loadSelected();
    });

    $("[value='Reset']").click(function () {
        $("#inputs").hide();
    });
 
});
