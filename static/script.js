
document.getElementById("send-btn").addEventListener("click", function() {
    sendMessage();
});

document.getElementById("user-input").addEventListener("keypress", function(e) {
    if (e.key === "Enter") {
        sendMessage();
    }
});

document.addEventListener("keydown", function(event) {
    if (event.keyCode === 13) {
function sendMessage() {
    var userInput = document.getElementById("user-input");
    var userMessage = userInput.value.trim();
    if (userMessage !== "") {
        var chatBody = document.getElementById("chat-body");
        var userChatMessage = document.createElement("div");
        userChatMessage.className = "chat-message";
        userChatMessage.innerHTML = '<div class="chat-message-inner user"><p>' + userMessage + '</p></div>';
        chatBody.appendChild(userChatMessage);
        userInput.value = "";
        userInput.focus();
        chatBody.scrollTop = chatBody.scrollHeight;
        setTimeout(function() {
            var assistantChatMessage = document.createElement("div");
            assistantChatMessage.className = "chat-message";
            assistantChatMessage.innerHTML = '<div class="chat-message-inner assistant"><p>Hi! How can I help you?</p></div>';
            chatBody.appendChild(assistantChatMessage);
            chatBody.scrollTop = chatBody.scrollHeight;
        }, 1000);
    }
}

        // Code to be executed when enter key is pressed
        // e.g., form submission or event triggering
    }
});


function populateQuestion() {
    // Set the value of the user-input field
    document.getElementById('user-input').value = 'Your question for Pitch Creator';
    // Submit the form
    document.getElementById('send-btn').click();
  }

  function populateQuestion1() {
    // Set the value of the user-input field
    document.getElementById('user-input').value = 'Your question for Blurb Creator';
    // Submit the form
    document.getElementById('send-btn').click();
  }
