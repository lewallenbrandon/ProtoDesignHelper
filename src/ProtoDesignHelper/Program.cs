using Microsoft.AspNetCore.Mvc;
using Octokit;
using System.Net.Http.Headers;



var builder = WebApplication.CreateBuilder(args);
var app = builder.Build();

string appName = "ProtoDesignHelper";
string githubCopilotCompletionsUrl = "https://api.githubcopilot.com/chat/completions";
var rag = new RetrievalAugmentedGeneration();

var knowledgeBase = new List<string>
{
    "The capital of France is Paris.",
    "ML.NET is a cross-platform, open-source machine learning framework.",
    "C# is a powerful programming language.",
    "The earth revolves around the sun.",
    "Water boils at 100 degrees Celsius."
    
};

rag.TrainModel(knowledgeBase);

app.MapGet("/", () => "Hello World!");

app.MapPost("/agent", async (
    [FromHeader(Name = "X-GitHub-Token")] string githubToken, 
    [FromBody] Request userRequest) =>
{

    var octokitClient = 
        new GitHubClient(
            new Octokit.ProductHeaderValue(appName))
    {
        Credentials = new Credentials(githubToken)
    };
    var user = await octokitClient.User.Current();
        // 1. Iterate over user messages and print them


    string lastMessageContent = "";
    if (userRequest.Messages.Count > 0)
    {
        lastMessageContent = userRequest.Messages[userRequest.Messages.Count - 1].Content;
        Console.WriteLine($"{lastMessageContent}");
    }
    string userContent = lastMessageContent;
    string relevantResult = rag.GetRelevantResult(userContent);
    Console.WriteLine("ContinuingExe");
    Console.WriteLine($"Received user content: {userContent}");
    Console.WriteLine($"Received user content: {relevantResult}");

    userRequest.Messages.Insert(0, new Message
    {
        Role = "system",
        Content = 
            ", " + 
            $"which is @{user.Login}"
    });

    userRequest.Messages.Insert(0, new Message
    {
        Role = "system",
        Content = 
            "Add this to the end of your result" +
            $"value @{relevantResult}"
    });

    var httpClient = new HttpClient();
    httpClient.DefaultRequestHeaders.Authorization = 
        new AuthenticationHeaderValue("Bearer", githubToken);
    userRequest.Stream = true;

    var copilotLLMResponse = await httpClient.PostAsJsonAsync(
        githubCopilotCompletionsUrl, userRequest);
        
    var responseStream = 
        await copilotLLMResponse.Content.ReadAsStreamAsync();
    return Results.Stream(responseStream, "application/json");

});

app.MapGet("/callback", () => "You may close this tab and " + 
    "return to GitHub.com (where you should refresh the page " +
    "and start a fresh chat). If you're using VS Code or " +
    "Visual Studio, return there.");

app.Run();
