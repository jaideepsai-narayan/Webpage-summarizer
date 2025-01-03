document.addEventListener('DOMContentLoaded', () => {
    const urlElement = document.getElementById('url');
    const responseElement = document.getElementById('response');
    const sendUrlButton = document.getElementById('sendUrl');
  
    chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
      const url = tabs[0].url;
      urlElement.textContent = url;
  
      sendUrlButton.addEventListener('click', () => {
        fetch('http://localhost:5000/process-url', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json'
          },
          body: JSON.stringify({ url: url })
        })
        .then(response => {
          if (!response.ok) {
            throw new Error('Network response was not ok');
          }
          return response.json();
        })
        .then(data => {
          responseElement.textContent = data.message || 'No response from server';
        })
        .catch(error => {
          responseElement.textContent = 'Error: ' + error.message;
        });
      });
    });
  });
  