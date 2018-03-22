$(document).ready(function () {
    // Init
    $('.loader').hide();
    $('#debug-out').hide();


    $('#btn-gen-anime').click(function () {
        var form_data = new FormData($('#upload-file')[0]);

        $(this).hide();
        $('.loader').show();

        $.ajax({
            type: 'POST',
            url: '/gen-anime',
            data: form_data,
            contentType: false,
            cache: false,
            processData: false,
            async: true,
            success: function (data) {
                // Get and display the result
                $('.loader').hide();
                $('#debug-out').fadeIn(600);
                $('#debug-out').text(' Result:  ' + data);
                console.log('Success!');
            },
        });
    });

});
