package main

import (
	"bufio"
	"context"
	"fmt"
	"os"

	"github.com/anthropics/anthropic-sdk-go"
)

func main() {
	// Initialize the Anthropics client. The Client is responsible for making requests to the Anthropics API.
	// NOTE: Make sure to set the ANTHROPIC_API_KEY environment variable
	client := anthropic.NewClient()

	// Create a new scanner to read user input
	scanner := bufio.NewScanner(os.Stdin)
	getUserMessage := func() (string, bool) {
		if !scanner.Scan() {
			return "", false
		}
		return scanner.Text(), true
	}

	// Create a new agent with the Anthropics client and the user message function
	agent := NewAgent(&client, getUserMessage)

	// Run the agent in a context. The context is used to manage the lifecycle of the request.
	err := agent.Run(context.TODO())
	if err != nil {
		fmt.Printf("Error: %s\n", err.Error())
	}
}

func NewAgent(client *anthropic.Client, getUserMessage func() (string, bool)) *Agent {
	/*
		Initialize a new agent with the Anthropics client and the user message function.
		The client is used to make requests to the Anthropics API, and the getUserMessage function
		is used to get user input.
		The agent is responsible for managing the conversation with the Anthropics API and
		handling the user input and output.

		Returns:
			*Agent: A new agent with the Anthropics client and the user message function.
	*/
	return &Agent{
		client:         client,
		getUserMessage: getUserMessage,
	}
}

type Agent struct {
	client         *anthropic.Client
	getUserMessage func() (string, bool)
}

func (a *Agent) Run(ctx context.Context) error {
	/*
		Params:
		- ctx: The context to use for the request. The context is used to manage the lifecycle of the request.
		Returns:
		- error: An error if the request fails, nil otherwise.
	*/
	conversation := []anthropic.MessageParam{}

	fmt.Println("Chat with Claude (use 'ctrl-c' to quit)")

	// Loop to get user input and send it to the Anthropics API
	// The loop continues until the user quits or an error occurs.
	for {
		fmt.Print("\u001b[94mYou\u001b[0m: ")
		userInput, ok := a.getUserMessage()
		if !ok {
			break
		}

		// Build a userMessage with the user input and add it to the conversation
		userMessage := anthropic.NewUserMessage(anthropic.NewTextBlock(userInput))
		conversation = append(conversation, userMessage)

		message, err := a.runInference(ctx, conversation)
		if err != nil {
			return err
		}
		conversation = append(conversation, message.ToParam())

		for _, content := range message.Content {
			switch content.Type {
			case "text":
				fmt.Printf("\u001b[93mClaude\u001b[0m: %s\n", content.Text)
			}
		}
	}

	return nil
}

func (a *Agent) runInference(ctx context.Context, conversation []anthropic.MessageParam) (*anthropic.Message, error) {
	message, err := a.client.Messages.New(ctx, anthropic.MessageNewParams{
		Model:     anthropic.ModelClaude3_7SonnetLatest, // Define the model to use for inference
		MaxTokens: int64(1024),                          // Define the maximum number of tokens to generate
		Messages:  conversation,                         // Send the conversation history to Claude's model
	})
	return message, err
}
