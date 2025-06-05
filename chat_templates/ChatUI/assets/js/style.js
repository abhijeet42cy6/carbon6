document.addEventListener('DOMContentLoaded', function () {
    const navbar = document.querySelector('.navbar');
    const dropdown = document.querySelector('.dropdown');
    const dropdownMenu = document.querySelector('#toggle-close')

    dropdownMenu.addEventListener('mouseenter', function () {
        dropdown.style.opacity = '0';
        dropdown.style.transition = 'opacity .5s';
        setTimeout(() => {
            dropdown.style.display = 'none';
        }, 300);
    })

    navbar.addEventListener('mouseenter', function () {
        dropdown.style.display = 'flex';
        dropdown.style.opacity = '0';
        setTimeout(() => {
            dropdown.style.opacity = '1';
            dropdown.style.transition = 'opacity 0.5s';
        }, 0);
    });

    navbar.addEventListener('mouseleave', function () {
        dropdown.style.opacity = '0';
        dropdown.style.transition = 'opacity .5s';
        setTimeout(() => {
            dropdown.style.display = 'none';
        }, 300);
    });
});
function hoverEffect(element) {
    const img = element.querySelector('.pic-inside');
    img.style.transform = 'translate(-50%, -50%) scale(1.1)'; // Slightly enlarge the image
}

function removeEffect(element) {
    const img = element.querySelector('.pic-inside');
    img.style.transform = 'translate(-50%, -50%) scale(1)'; // Reset to original size
}
function moveBackground(selectedDiv) {
    const background = document.querySelector('.background');
    const offsetTop = selectedDiv.offsetTop;
    const height = selectedDiv.offsetHeight;
    background.style.transform = `translateY(${offsetTop}px)`;
    background.style.height = `${height}px`;
}

window.onload = function () {
    const firstChild = document.querySelector('.child');
    moveBackground(firstChild);
};








document.addEventListener("DOMContentLoaded", function() {
    // Initialize the default image and background position
    const defaultImage = "./assets/images/backgrounds/diff.png";
    document.querySelector(".image-container img").src = defaultImage;
    
    // Initialize the background position for the first child
    const firstChild = document.querySelector('.child');
    moveBackground(firstChild);
  });
  
  function moveBackground(element) {
    // 1. Move the sliding background highlight
    const background = document.querySelector('.background');
    const offsetTop = element.offsetTop;
    const height = element.offsetHeight;
    
    background.style.transform = `translateY(${offsetTop}px)`;
    background.style.height = `${height}px`;
  
    // 2. Change the image
    const imagePath = element.getAttribute("data-image");
    const imageContainer = document.querySelector(".image-container img");
    imageContainer.src = imagePath;
  
    // 3. Update active state
    document.querySelectorAll(".child").forEach(child => child.classList.remove("selected"));
    element.classList.add("selected");
  }