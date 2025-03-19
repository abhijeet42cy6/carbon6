// Store chats in localStorage
let chats = JSON.parse(localStorage.getItem('chats')) || [];
let currentChatIndex = -1;
let editingChatIndex = -1;
let deletingChatIndex = -1;
let cameraStream = null;

// Initialize the application
document.addEventListener('DOMContentLoaded', () => {
    loadChats();
    setupEventListeners();
    
    // Load last active chat if exists
    const lastActiveChat = localStorage.getItem('lastActiveChat');
    if (lastActiveChat !== null) {
        currentChatIndex = parseInt(lastActiveChat);
        if (currentChatIndex >= 0 && currentChatIndex < chats.length) {
            showChat(currentChatIndex);
        }
    }
});

function loadChats() {
    const chatList = document.getElementById('chatList');
    chatList.innerHTML = '';
    
    chats.forEach((chat, index) => {
        const chatElement = createChatElement(chat, index);
        chatList.appendChild(chatElement);
    });
}

function createChatElement(chat, index) {
    const div = document.createElement('div');
    div.className = 'chat-item d-flex justify-content-between align-items-center';
    div.onclick = () => switchChat(index);
    div.innerHTML = `
        <span>${chat.name}</span>
        <div class="chat-options">
            <button class="btn btn-sm btn-outline-secondary me-1" onclick="showEditModal(${index}, event)">
                <i class="bi bi-pencil"></i>
            </button>
            <button class="btn btn-sm btn-outline-danger" onclick="showDeleteModal(${index}, event)">
                <i class="bi bi-trash"></i>
            </button>
        </div>
    `;
    return div;
}

function setupEventListeners() {
    // Modal related
    document.getElementById('saveChatNameBtn').addEventListener('click', saveEditedChatName);
    document.getElementById('confirmDeleteBtn').addEventListener('click', confirmDeleteChat);
    document.getElementById('takePictureBtn').addEventListener('click', takePicture);
    
    // Other listeners
    document.getElementById('newChatBtn').addEventListener('click', createNewChat);
    document.getElementById('sendButton').addEventListener('click', sendMessage);
    document.getElementById('chatInput').addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            sendMessage();
        }
    });
    document.getElementById('documentInput').addEventListener('change', handleFileSelected);
    document.getElementById('imageInput').addEventListener('change', handleFileSelected);

    // Handle camera modal close
    document.getElementById('cameraModal').addEventListener('hidden.bs.modal', () => {
        if (cameraStream) {
            cameraStream.getTracks().forEach(track => track.stop());
            cameraStream = null;
        }
    });
}

function createNewChat() {
    const chatName = `New Chat`;
    chats.push({
        name: chatName,
        messages: []
    });
    currentChatIndex = chats.length - 1;
    saveChats();
    loadChats();
    showChat(currentChatIndex);
}

function showEditModal(index, event) {
    event.stopPropagation();
    editingChatIndex = index;
    const modal = new bootstrap.Modal(document.getElementById('editChatModal'));
    document.getElementById('editChatInput').value = chats[index].name;
    modal.show();
}

function saveEditedChatName() {
    const newName = document.getElementById('editChatInput').value.trim();
    if (newName && editingChatIndex !== -1) {
        chats[editingChatIndex].name = newName;
        saveChats();
        loadChats();
        bootstrap.Modal.getInstance(document.getElementById('editChatModal')).hide();
    }
}

function showDeleteModal(index, event) {
    event.stopPropagation();
    deletingChatIndex = index;
    const modal = new bootstrap.Modal(document.getElementById('deleteChatModal'));
    modal.show();
}

function confirmDeleteChat() {
    if (deletingChatIndex !== -1) {
        chats.splice(deletingChatIndex, 1);
        if (currentChatIndex === deletingChatIndex) {
            currentChatIndex = -1;
            showWelcomeSection();
        } else if (currentChatIndex > deletingChatIndex) {
            currentChatIndex--;
        }
        saveChats();
        loadChats();
        bootstrap.Modal.getInstance(document.getElementById('deleteChatModal')).hide();
    }
}

function switchChat(index) {
    currentChatIndex = index;
    localStorage.setItem('lastActiveChat', index);
    showChat(index);
}

function showChat(index) {
    const messagesContainer = document.getElementById('messagesContainer');
    const welcomeSection = document.querySelector('.welcome-section');
    
    if (chats[index].messages.length === 0) {
        // Show welcome section for new chats
        if (welcomeSection) {
            welcomeSection.style.display = 'block';
        }
        messagesContainer.innerHTML = '';
    } else {
        // Hide welcome section and show messages
        if (welcomeSection) {
            welcomeSection.style.display = 'none';
        }
        messagesContainer.innerHTML = '';
        chats[index].messages.forEach(message => {
            const messageElement = document.createElement('div');
            messageElement.className = `message ${message.type}-message mb-3`;
            messageElement.innerHTML = `
                <div class="d-flex ${message.type === 'user' ? 'justify-content-end' : 'justify-content-start'}">
                    <div class="message-content p-3">
                        ${message.content}
                    </div>
                </div>
            `;
            messagesContainer.appendChild(messageElement);
        });
    }
    
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
}

function showWelcomeSection() {
    const messagesContainer = document.getElementById('messagesContainer');
    const welcomeSection = document.querySelector('.welcome-section');
    
    if (welcomeSection) {
        welcomeSection.style.display = 'block';
    }
    messagesContainer.innerHTML = '';
}

function saveChats() {
    localStorage.setItem('chats', JSON.stringify(chats));
}

// Helper function to add messages to UI
function addMessageToUI(message, className) {
    const messagesContainer = document.getElementById('messagesContainer');
    const welcomeSection = document.querySelector('.welcome-section');
    
    if (welcomeSection) {
        welcomeSection.style.display = 'none';
    }
    
    const messageElement = document.createElement('div');
    messageElement.className = `message ${className} mb-3`;
    messageElement.innerHTML = `
        <div class="d-flex ${className === 'user-message' ? 'justify-content-end' : 'justify-content-start'}">
            <div class="message-content">
                ${message}
            </div>
        </div>
    `;
    messagesContainer.appendChild(messageElement);
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
}

// Updated sendMessage function
async function sendMessage() {
    const input = document.getElementById('chatInput');
    const message = input.value.trim();
    console.log("sendMessage function triggered!");

    if (message && currentChatIndex !== -1) {
        // Update chat name if this is the first message
        if (chats[currentChatIndex].messages.length === 0) {
            const words = message.split(' ').slice(0, 3).join(' ');
            chats[currentChatIndex].name = words + (message.split(' ').length > 3 ? '...' : '');
        }

        // Add message to chat storage
        chats[currentChatIndex].messages.push({
            type: 'user',
            content: message,
            timestamp: new Date().toISOString()
        });
        
        // Save chats and update UI
        saveChats();
        loadChats();
        
        // Add user message to UI
        addMessageToUI(message, 'user-message');
        
        // Clear input 
        input.value = '';

        // Send request to the FastAPI backend
        try {
            const response = await fetch('http://localhost:8000/chat', {
                method: 'POST',
                headers: {
                    'Accept': 'application/json',
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ query: message })
            });

            if (response.ok) {
                const data = await response.json();
                const botResponse = data.response || "No response from server";
                
                // Add bot response to chat storage
                chats[currentChatIndex].messages.push({
                    type: 'bot',
                    content: botResponse,
                    timestamp: new Date().toISOString()
                });
                
                // Save chats and update UI
                saveChats();
                addMessageToUI(botResponse, 'bot-message');
            } else {
                const errorMessage = "Error communicating with server";
                addMessageToUI(errorMessage, 'bot-message');
                
                // Add error message to chat storage
                chats[currentChatIndex].messages.push({
                    type: 'bot',
                    content: errorMessage,
                    timestamp: new Date().toISOString()
                });
                saveChats();
            }
        } catch (error) {
            const errorMessage = "Failed to reach server";
            addMessageToUI(errorMessage, 'bot-message');
            
            // Add error message to chat storage
            chats[currentChatIndex].messages.push({
                type: 'bot',
                content: errorMessage,
                timestamp: new Date().toISOString()
            });
            saveChats();
            console.error("Error:", error);
        }
    }
}

function fillQuery(element) {
    const input = document.getElementById('chatInput');
    input.value = element.textContent.trim();
    input.focus();
}

function handleFileUpload(type) {
    const input = type === 'document' ? 
        document.getElementById('documentInput') : 
        document.getElementById('imageInput');
    input.click();
}

async function handleFileSelected(event) {
    const file = event.target.files[0];
    if (file && currentChatIndex !== -1) {
        // Add file message to chat
        chats[currentChatIndex].messages.push({
            type: 'user',
            content: `Uploaded: ${file.name}`,
            timestamp: new Date().toISOString()
        });
        
        saveChats();
        
        // Display file message
        const messagesContainer = document.getElementById('messagesContainer');
        const messageElement = document.createElement('div');
        messageElement.className = 'message user-message mb-3';
        messageElement.innerHTML = `
            <div class="d-flex justify-content-end">
                <div class="message-content p-3">
                    Uploaded: ${file.name}
                </div>
            </div>
        `;
        messagesContainer.appendChild(messageElement);
        event.target.value = '';

        // Prepare FormData to send file
        const formData = new FormData();
        formData.append("files", file);  // "files" matches FastAPI's parameter

        try {
            const response = await fetch("http://52.5.167.111:8001/upload", {
                method: "POST",
                body: formData
            });

            const result = await response.json();
            if (response.ok) {
                console.log("File uploaded successfully:", result.message);
            } else {
                console.error("File upload failed:", result.error);
            }
        } catch (error) {
            console.error("Error uploading file:", error);
        }
    }
}


function openCamera() {
    const modal = new bootstrap.Modal(document.getElementById('cameraModal'));
    const video = document.getElementById('cameraPreview');
    
    if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
        navigator.mediaDevices.getUserMedia({ video: true })
            .then(function(stream) {
                cameraStream = stream;
                video.srcObject = stream;
                modal.show();
            })
            .catch(function(error) {
                console.error("Error accessing camera:", error);
                alert("Error accessing camera. Please make sure you have granted camera permissions.");
            });
    } else {
        alert("Sorry, your browser doesn't support camera access");
    }
}

function takePicture() {
    if (cameraStream && currentChatIndex !== -1) {
        const video = document.getElementById('cameraPreview');
        const canvas = document.createElement('canvas');
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        canvas.getContext('2d').drawImage(video, 0, 0);
        
        // Convert to data URL and add to chat
        const imageUrl = canvas.toDataURL('image/jpeg');
        chats[currentChatIndex].messages.push({
            type: 'user',
            content: `<img src="${imageUrl}" style="max-width: 200px; border-radius: 4px;">`,
            timestamp: new Date().toISOString()
        });
        
        saveChats();
        showChat(currentChatIndex);
        
        // Close modal and stop camera
        bootstrap.Modal.getInstance(document.getElementById('cameraModal')).hide();
    }
}
