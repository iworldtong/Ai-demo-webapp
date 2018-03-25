$(document).ready(function () {
    // Init
    $('.image-section').hide();
    $('.loader').hide();
    $('#debug-out').hide();


    $('#btn-gen-anime').click(function () {
        var img_preview_obj = document.getElementById("imagePreview");

        $('#btn-gen-anime').hide();
        $('#debug-out').hide();
        $('.loader').show();

        $.ajax({
            type: 'POST',
            url: '/gen-anime',
            //data: form_data,
            contentType: false,
            cache: false,
            processData: false,
            async: true,
            success: function (img_stream) {                                
                $('.loader').hide();                
                $('#btn-gen-anime').show();                
                $('#imagePreview').css('background-image', 'url(data:image/jpg;base64,' + img_stream + ')');                
                $('.image-section').hide();  
                $('.image-section').fadeIn(600);                                
                $('#debug-out').fadeIn(600);                
                $('#debug-out').text('OK');
                console.log('Success!');
            },
        });
    });

});
