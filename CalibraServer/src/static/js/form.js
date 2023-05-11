$(document).ready(function(){
    
    var current_fs, next_fs, previous_fs;
    var opacity;

    $(".next").click(function(){
        $("#progressbar li.active").next().addClass("active");
        current_fs = $(this).parent();
        next_fs = $(this).parent().next();
        next_fs.show();
        current_fs.animate({opacity: 0}, {
            step: function(now) {
                opacity = 1 - now;
                current_fs.css({
                    'display': 'none',
                    'position': 'relative'
                });
                next_fs.css({'opacity': opacity});
            },
            duration: 600
        });
    });

    $(".previous").click(function(){
        $("#progressbar li.active").last().removeClass("active");
        current_fs = $(this).parent();
        previous_fs = $(this).parent().prev();
        previous_fs.show();
        current_fs.animate({opacity: 0}, {
            step: function(now) {
                opacity = 1 - now;
                current_fs.css({
                    'display': 'none',
                    'position': 'relative'
                });
                previous_fs.css({'opacity': opacity});
            },
            duration: 600
        });
    });

    $("#next_btn_pipeline").prop("disabled", true);
    $("#next_btn_computing_infra").prop("disabled", true);
    $("#next_btn_network_infra").prop("disabled", true);

    // SETTINGS STEP
    const populationSlider = document.getElementById('population_range');
    const populationValue = document.getElementById('population_value');
    populationSlider.addEventListener('input', (event) => {
      populationValue.textContent = event.target.value;
    });

    const generationsCheck = document.getElementById('generations_check');
    const generationsSlider = document.getElementById('generations_range');
    const generationsValue = document.getElementById('generations_value');
    generationsSlider.addEventListener('input', (event) => {
      generationsValue.textContent = event.target.value;
    });

    const timeCheck = document.getElementById('time_check');
    const timeSlider = document.getElementById('time_range');
    const timeValue = document.getElementById('time_value');
    timeSlider.addEventListener('input', (event) => {
      timeValue.textContent = event.target.value;
    });

    // Add event listeners for the mouse wheel event
    window.addEventListener('wheel', handleWheelEvent);
    function handleWheelEvent(event) {
      if (event.target.type === 'range') {
        const rangeElement = event.target;
        console.log(rangeElement);
        const valueElement = rangeElement.previousElementSibling;
        console.log(valueElement);

        let currentValue = parseInt(rangeElement.value);
        if (event.deltaY < 0) {
          currentValue += parseInt(rangeElement.step);
        } else {
          currentValue -= parseInt(rangeElement.step);
        }

        currentValue = Math.max(parseInt(rangeElement.min), Math.min(parseInt(rangeElement.max), currentValue));
        rangeElement.value = currentValue;
        console.log(currentValue.toString());
        valueElement.textContent = currentValue.toString();
      }
    }

    generationsCheck.addEventListener('click', () => {
      if (generationsCheck.checked){
          generationsSlider.disabled = false;
      }else{
          generationsSlider.disabled = true;
      }
    });

    timeCheck.addEventListener('click', () => {
      if (timeCheck.checked){
          timeSlider.disabled = false;
      }else{
          timeSlider.disabled = true;
      }
    });

    console.clear();
    ('use strict');

    // Drag and drop - single or multiple image files
    // https://www.smashingmagazine.com/2018/01/drag-drop-file-uploader-vanilla-js/
    // https://codepen.io/joezimjs/pen/yPWQbd?editors=1000


    'use strict';

    // Four objects of interest: drop zones, input elements, gallery elements, and the files.
     // dataRefs = {files: [image files], input: element ref, gallery: element ref}
    const preventDefaults = event => {
      event.preventDefault();
      event.stopPropagation();
    };

    const highlight = event =>
        event.target.classList.add('highlight');

    const unhighlight = event =>
        event.target.classList.remove('highlight');

    const getInputAndGalleryRefs = element => {
        const zone = element.closest('.upload_dropZone') || false;
        const gallery = zone.querySelector('.upload_gallery') || false;
        const input = zone.querySelector('input[type="file"]') || false;
        return {input: input, gallery: gallery};
    }

    const eventHandlers = zone => {
        const dataRefs = getInputAndGalleryRefs(zone);
        if (!dataRefs.input) return;
        // Prevent default drag behaviors
        ;['dragenter', 'dragover', 'dragleave', 'drop'].forEach(event => {
        zone.addEventListener(event, preventDefaults, false);
        document.body.addEventListener(event, preventDefaults, false);
        });
        // Highlighting drop area when item is dragged over it
        ;['dragenter', 'dragover'].forEach(event => {
            zone.addEventListener(event, highlight, false);
        });
        ;['dragleave', 'drop'].forEach(event => {
            zone.addEventListener(event, unhighlight, false);
        });
    }

    // Initialise ALL dropzones
    const dropZones = document.querySelectorAll('.upload_dropZone');
    for (const zone of dropZones) {
        eventHandlers(zone);
        if (zone.classList.contains('csv')){
            // Handle browse selected files
            const dataRefs = getInputAndGalleryRefs(zone);
            var parentDiv = zone.parentNode;
            var nextButton = parentDiv.nextElementSibling.nextElementSibling;
            dataRefs.input.addEventListener('change', event => {
                dataRefs.files = event.target.files;
                handleCSVFiles(dataRefs);
            }, false);
            // Handle dropped files
            zone.addEventListener('drop', event => {
                dataRefs.files = [event.dataTransfer.files[0]];
                handleCSVFiles(dataRefs);
            }, false);
        }else if(zone.classList.contains('yml')){
            const dataRefs = getInputAndGalleryRefs(zone);
            dataRefs.input.addEventListener('change', event => {
                dataRefs.files = event.target.files;
                handleYMLFiles(dataRefs);
            }, false);
            zone.addEventListener('drop', event => {
                const dataRefs = getInputAndGalleryRefs(event.target);
                dataRefs.files = [event.dataTransfer.files[0]];
                handleYMLFiles(dataRefs);
            }, false);
        }
    }
    // Double checks the input "accept" attribute
    const isCSVFile = file => file.type === 'text/csv';
    const isYMLFile = file => file.name.endsWith('.yml') || file.name.endsWith('.yaml')

    function previewFiles(dataRefs) {
        if (!dataRefs.gallery) return;
        for (const file of dataRefs.files) {
            let reader = new FileReader();
            reader.readAsDataURL(file);
            reader.onloadend = function() {
                let span = document.createElement('span');
                let img = document.createElement('img');
                img.src = '/static/img/file.png';
                img.height = (30);
                img.width = (30);
                img.alt = 'File icon';
                span.appendChild(img);
                span.appendChild(document.createTextNode(file.name));
                while (dataRefs.gallery.firstChild) {
                    dataRefs.gallery.removeChild(dataRefs.gallery.firstChild);
                }
                dataRefs.gallery.appendChild(span);
                console.log(dataRefs)
            }
        }
    }

  // Based on: https://flaviocopes.com/how-to-upload-files-fetch/
  const fileUpload = dataRefs => {

    // Multiple source routes, so double check validity
    if (!dataRefs.files || !dataRefs.input) return;

    const url = dataRefs.input.getAttribute('data-post-url');
    if (!url) return;

    const name = dataRefs.input.getAttribute('data-post-name');
    if (!name) return;

    const formData = new FormData();
    formData.append(name, dataRefs.files);

    fetch(url, {
      method: 'POST',
      body: formData
    })
    .then(response => response.json())
    .then(data => {
      console.log('posted: ', data);
      if (data.success === true) {
        previewFiles(dataRefs);
      } else {
        console.log('URL: ', url, '  name: ', name)
      }
    })
    .catch(error => {
      console.error('errored: ', error);
    });
  }


  // Handle both selected and dropped files
  const handleCSVFiles = dataRefs => {
    let files = [...dataRefs.files];
    files = files.filter(item => {
      if (!isCSVFile(item)) {
        console.log('Not CSV file, ', item.type);
      }else{
          var parentDiv = dataRefs.input.parentNode.parentNode;
          var nextButton = parentDiv.nextElementSibling.nextElementSibling;
          if(nextButton.getAttribute("id") === 'next_btn_computing_infra'){
              next_btn_computing_infra.disabled = 0;
          }else{
              next_btn_network_infra.disabled = 0;
          }
      }
      return isCSVFile(item) ? item : null;
    });
    if (!files.length) return;
    dataRefs.files = [files[0]];
    previewFiles(dataRefs);
    fileUpload(dataRefs);
  }
  const handleYMLFiles = dataRefs => {
    let files = [...dataRefs.files];
    // Remove unaccepted file types
    files = files.filter(item => {
      if (!isYMLFile(item)) {
        console.log('Not YAML file, ', item.type);
      }else{
          next_btn_pipeline.disabled = 0;
      }
      return isYMLFile(item) ? item : null;
    });
    if (!files.length) return;
    dataRefs.files = [files[0]];
    previewFiles(dataRefs);
    fileUpload(dataRefs);
  }


});