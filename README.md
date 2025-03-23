# HandSpeak_AI
Project for GenAI Genesis 2025 Hackathon

## Inspiration

The inspiration behind HandSpeak AI came from the communication barrier between individuals who use sign language and those who do not understand it, with the goal of enabling communication between everyone. Additionally, the project is not limited to sign language alone, but can also incorporate any gesture related to a specific meaning, expanding its reach to more inclusive communication.

## What it does

**HandSpeak** AI translates sign language gestures into text and audio in real-time. With this project, individuals who communicate through sign language can easily interact with those who do not understand it, fostering inclusive communication. The system works by using a USB camera to capture the gestures and a Bluetooth speaker to output the voice translation, both on computers and the Jetson Orin Nano platform. The latter was implemented to make the project more portable by leveraging modern technologies.

## How we built it

The project is entirely built using Python 3. To use it, only a platform with a camera and audio output is required. Setting up the configuration also requires a monitor, keyboard, and mouse. Additionally, PyTorch is used, which may require specific version compatibility.

## Challenges we ran into

We faced a lot of trial and error during the development of the project. One of the biggest challenges was setting up the entire Python library environment on the Jetson Orin Nano platform, mainly due to version compatibility issues between the system and its components. On the other hand, gesture detection—both the training process and feature extraction necessary for accurate gesture recognition—was a significant challenge. While it's not fully resolved, it works quite well with proper training.

## Accomplishments that we're proud of

In the end, we achieved our goal. Although we didn't implement the entire sign language or complex phrases, we were able to create a working proof of concept for the idea.

## What we learned

hroughout the development of HandSpeak AI, we learned valuable lessons about the complexities of gesture recognition, the importance of optimizing system compatibility across different platforms, and the challenges of ensuring accurate translations in real-time. We also gained insights into how the Jetson Orin Nano works, as this was our second AI project and the first project on this platform, which was recently acquired by the lab where we are currently doing our internship.

## What's next for HandSpeak AI

Next steps include optimizing the program, creating a comprehensive dataset to incorporate basic gestures, and adding new features to the GUI to enhance the user experience. Additionally, we plan to improve the accuracy of gesture detection.
