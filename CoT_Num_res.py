import os
import torch
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from unsloth import FastLanguageModel
import inflect

# Step 1: Set CUDA_VISIBLE_DEVICES to mask only GPU 1
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# Step 2: Set the device to 'cuda:0' since GPU 1 is now 'cuda:0' within this environment
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
print(device)

# Initialize inflect engine for text-to-number conversion
p = inflect.engine()

fourbit_models = [
    "unsloth/Meta-Llama-3.1-8B-bnb-4bit",      # Llama-3.1 15 trillion tokens model 2x faster!
    "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
    "unsloth/Meta-Llama-3.1-70B-bnb-4bit",
    "unsloth/Meta-Llama-3.1-405B-bnb-4bit",    # We also uploaded 4bit for 405b!
    "unsloth/Mistral-Nemo-Base-2407-bnb-4bit", # New Mistral 12b 2x faster!
    "unsloth/Mistral-Nemo-Instruct-2407-bnb-4bit",
    "unsloth/mistral-7b-v0.3-bnb-4bit",        # Mistral v3 2x faster!
    "unsloth/mistral-7b-instruct-v0.3-bnb-4bit",
    "unsloth/Phi-3.5-mini-instruct",           # Phi-3.5 2x faster!
    "unsloth/Phi-3-medium-4k-instruct",
    "unsloth/gemma-2-9b-bnb-4bit",
    "unsloth/gemma-2-27b-bnb-4bit",            # Gemma 2x faster!
]
# Load the model and tokenizer
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Mistral-Nemo-Instruct-2407",  # Use your selected model
    max_seq_length=5000,
    load_in_4bit=True  # Enable 4-bit quantization for memory efficiency
)

model.eval()  # Set to eval mode

# Load the dataset
test_data = pd.read_csv("train_num_reas.csv")

# Define custom Dataset
class CoTTestDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=1000):  # MATCHED max_length
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        headline = row["masked headline"]
        news_content = row["news"]

        # Chain-of-Thought prompt
        input_text = (
            f"### News Article:\n{news_content}\n\n"
            f"### Headline with a Blank:\n'{headline}'\n\n"
            "### Goal:\n"
            "Fill in the blank in the headline with a number based on the news article.\n"
            "Use the following reasoning steps:\n"
            "1. Extract relevant numbers from the article.\n"
            "2. Determine if the number needs to be copied, paraphrased, or calculated.\n"
            "3. If calculation is required, perform the necessary arithmetic operations.\n"
            "4. Provide the final answer that best fits the headline.\n\n"
            "### Answer: <Provide the answer that fills the blank>\n"

            "### Example 1:\n"
            "News Article:\n"
                "A burglar made a huge mistake yesterday when he decided to break into the Manhattan apartment of a computer geek who had set up a motion-detecting camera that transmitted photos directly to his Smartphone, reports the New York Post. Levent Cetiner was six blocks from his home when he received an email showing a thief looting his stuff. Cetiner called 911 and raced home where he banged on the door and yelled, You are being recorded, and the police are on the way! The cops arrived, and Cetiner showed them the footage while the burglar was still inside. Eventually the bandit fled by fire escape and was nabbed in a nearby courtyard by police \n\n"
            "Headline with a Blank:\n"
            "'IT Guy Foils Burglar From ____ Blocks Away'\n\n"
            "Reasoning:\n"
            "Explanations: The news article mentions that the computer geek, Levent Cetiner, was six blocks from his home when he received the email with the footage of the burglar. Therefore, the number that best fits the headline is six \n"
            "Answer:\n"
            "6\n\n"

"### Example 2:\n"
            "News Article:\n"
                "Oregon's incredible offense busted up Wisconsin and the record books on the way to the Ducks' first Rose Bowl victory in 95 years. Darron Thomas passed for three touchdowns, De'Anthony Thomas scored on runs of 91 and 64 yards, and the No. 6 Ducks earned their first bowl victory under coach Chip Kelly, holding off Wisconsin 45-38 tonight in the highest-scoring Rose Bowl ever played. And it wasn't over until a video review confirmed the Badgers (11-3) ran out of time at the Oregon 25, out of timeouts, and unable to spike the ball in time to stop the clock for a last-gasp fling. Lavasier Tuinei caught eight passes for 158 yards and two TDs for the Ducks (12-2), who had no postseason success to show for Kelly's otherwise wildly successful three-year tenure until this landmark offensive performance in the 98th Rose Bowl. Oregon hadn't won the West Coast's biggest game since 1917. The Granddaddy of Them All had never seen this many points, beating the record 80 scored by Washington and Iowa in 1991. With the Ducks wearing mirrored helmets and playing at their usual frantic pace, Oregon racked up 621 total yards, just shy of the Rose Bowl record. Click for more from the game. \n\n"
            "Headline with a Blank:\n"
            "'Oregon Wins ____st Rose Bowl in 95 Years'\n\n"
            "Reasoning:\n"
            "Explanations:The news article mentions that Oregon earned their first Rose Bowl victory in 95 years, making it clear that they had not won the game since 1917. So the number that fits in the blank is 1.  \n"
            "Answer:\n"
            "1\n\n"

"### Example 3:\n"
            "News Article:\n"
                "Stocks extended losses today off the news of a 3.8% decline in US GDP for the fourth quarter, ending a turbulent month, MarketWatch reports. The report capped a week of widespread layoffs and poor earnings at major US firms. Amazon.com was a rare exception, up 18% on strong fourth-quarter earnings. The Dow fell 148.15 to close at 8,000.86. The Nasdaq closed down 31.42 at 1,476.42, and the S&P 500 lost 19.26 to settle at 825.88.  \n\n"
            "Headline with a Blank:\n"
            "'Dow Off ____, Down 8.8% for Month '\n\n"
            "Reasoning:\n"
            "Explanations:The article mentions that the Dow fell 148.15 points by the end of the day. Since headlines usually round off numbers for simplicity, 148.15 becomes 148 when rounded to the nearest whole number. That’s the number that fits best in the blank.   \n"
            "Answer:\n"
            "148\n\n"

"### Example 4:\n"
            "News Article:\n"
                "Sporting a goatee and an old-school swimsuit, Michael Phelps picked up right where he left off in Beijing. Two races. Two wins. The world's greatest swimmer left a 9-month layoff and marijuana travails in his wake tonight at his first meet since a triumphant Olympics. None of it seemed to matter when Phelps dove in for two events less than an hour apart at the Charlotte UltraSwim in Charlotte, NC. He started with a victory in the 200-meter freestyle and came right back to touch first in the 100 butterfly, both with times that easily broke the meet records he set 3 years ago.   I was real happy with today,  Phelps said.  I think the training is working well.  Coach Bob Bowman, usually Phelps' harshest critic, was downright giddy when he saw the times—1:46 in the 200 free and 51.72 in the fly.   \n\n"
            "Headline with a Blank:\n"
            "'Phelps Snags ____ Wins at NC Meet '\n\n"
            "Reasoning:\n"
            "Explanations:The article says Phelps competed in two races — the 200-meter freestyle and the 100 butterfly — and won both. So he had two wins at the meet. The word “two” translates to the number 2, which fits in the blank.   \n"
            "Answer:\n"
            "2\n\n"

"### Example 5:\n"
            "News Article:\n"
                "A book reportedly written by Canadian serial killer Robert Pickton was removed from Amazon's website a day after being put on sale online following protests by authorities in British Columbia. Publisher Outskirts Press issued a statement saying it had asked Amazon to remove the book from its website and apologizing to the families of the victims for any additional heartache this may have caused. Pickton, now 66, was convicted in 2007 of six counts of second-degree murder in the deaths of sex workers. Pickton slaughtered the women at his pig farm and fed some remains to his pigs. He was sentenced to life in prison. Some 20 other murder charges were stayed, though DNA from 33 women was found on the farm and he told an undercover officer he had killed 49. By Monday afternoon, the 144-page memoir titled Pickton: In His Own Words was no longer available. CTV News reports that the handwritten manuscript, which proclaims Pickton's innocence, was apparently passed to one of Pickton's former cellmates in the maximum-security Kent Institution. Authorities in British Columbia promised to introduce a law to prevent offenders from profiting from their crimes. I am at a loss for words. To think about the pain that he's prepared to willingly cause all of the families of those people who he murdered, British Columbia Premier Christy Clark told reporters. I have trouble understanding it and I think people will want to know that their government is doing everything it can ... to stop him from profiting from this at the very least.   \n\n"
            "Headline with a Blank:\n"
            "'Amazon Pulls Serial Killer's Book After ____ Day  '\n\n"
            "Reasoning:\n"
            "Explanations:The article says the book was taken down a day after it was listed on Amazon. Since a day means one day, the number that fits best in the blank is 1   \n"
            "Answer:\n"
            "1\n\n"

"### Example 6:\n"
            "News Article:\n"
                "Guenter Grass, the Nobel-winning German writer who gave voice to the generation that came of age during the horrors of the Nazi era but later ran into controversy over his own World War II past and stance toward Israel, has died. He was 87. Grass was lauded by Germans for helping to revive their culture in the aftermath of World War II and helping to give voice and support to democratic discourse in the postwar nation. Yet he provoked the ire of many in 2006 when he revealed in his memoir that, as a teenager, he had served in the Waffen-SS, the combat arm of Adolf Hitler's notorious paramilitary organization. A trained sculptor, Grass made his literary reputation with The Tin Drum, published in 1959. Combining naturalistic detail with fantastical images, his work captured the German reaction to the rise of Nazism, the horrors of the war, and the guilt that lingered after Adolf Hitler's defeat. Three decades after its release, in 1999, the Swedish Academy honored Grass with the Nobel Prize for literature, praising him for setting out to revive German literature after the Nazi era. With The Tin Drum, the Nobel Academy said,  it was as if German literature had been granted a new beginning after decades of linguistic and moral destruction.    \n\n"
            "Headline with a Blank:\n"
            "' Nobel-Winning Author Guenter Grass Dies at ____  '\n\n"
            "Reasoning:\n"
            "Explanations: The article says Guenter Grass has died, and it mentions that he was 87 years old at the time of his death. That’s the number that fits in the blank.   \n"
            "Answer:\n"
            "87\n\n"

"### Example 6:\n"
            "News Article:\n"
                "GQ has a fascinating look at the little-known world of the  club-appearance economy,  which it calls  a bizarre ecosystem that has reinvented the way a famous person ... makes a living.  Simply put, clubs in Las Vegas, Miami, and around the world will pay people of varying levels of fame—Lil Jon, Puffy's son, Lenny Kravitz, Bravo's housewives, Tyson Beckford—exorbitant amounts of money to hang out for an hour. Sources say Future was recently paid $250,000 to appear at a club on New Year's Eve. Ray J got multiple $250,000 paydays when he threw himself four birthday parties at four different clubs in January. Even a lesser Kardashian like Scott Disick could bring in $80,000 just to walk into a club, sit at a VIP table, and have a drink for an hour. The blame for the advent of the club-appearance economy is laid at the feet of Paris Hilton— who made it possible to be famous for doing nothing —as one would expect it must be. Since then, the beneficiaries of the economy have shifted from actual celebrities, to reality stars, to DJs, to rappers, and finally to whatever you call someone with millions of Instagram followers. And while club appearances are an integral source of income for the Disicks of the world, they're even more important to the clubs, which can charge more than $25,000 per table just to be seated near a bought-and-paid-for celebrity. A Vegas club recently sued Nicki Minaj, who it claims came up 26 minutes short of her contract-mandated hour of club time. More than Minaj's $263,000 fee, the club was mostly concerned about the five-figure (per table) bottle service money it lost from those who would have sat adjacent to Minaj. Read the full story here.     \n\n"
            "Headline with a Blank:\n"
            "'  Why 'Celebs' Get $____K to Hang Out at a Club for an Hour'\n\n"
            "Reasoning:\n"
            "Explanations: The article says stars like Future and Ray J were paid $250,000 for a single club appearance. Since the headline uses “K” to shorten thousands, $250,000 becomes $250K. That’s the number that fits in the blank.\n"
            "Answer:\n"
            "250\n\n"
    
           
        )

        # Tokenize input
        inputs = self.tokenizer(
            input_text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

        return {key: val.squeeze(0) for key, val in inputs.items()}

# Create DataLoader
test_dataset = CoTTestDataset(test_data, tokenizer, max_length=5000)
test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)

# Prepare model for inference
FastLanguageModel.for_inference(model)

# Init result containers
predicted_numbers = []
predicted_texts = []
full_outputs = []

def convert_text_to_number(text):
    try:
        return str(p.number_to_words(text)) if p.number_to_words(text) else text
    except:
        return text

# Inference loop
with torch.no_grad():
    for batch in tqdm(test_loader, desc="Generating Predictions"):
        inputs = {key: val.to(device) for key, val in batch.items()}

        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            use_cache=True
        )

        decoded_preds = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        for pred in decoded_preds:
            full_outputs.append(pred)
            answer_text = ""

            for line in pred.splitlines():
                if line.startswith("### Answer:"):
                    answer_text = line.split(":", 1)[1].strip()
                    break

            predicted_texts.append(answer_text)
            converted_answer = convert_text_to_number(answer_text)
            predicted_numbers.append(converted_answer)

# Add results to DataFrame
results_df = pd.DataFrame({
    "predicted_ans": predicted_numbers,
    "predicted_text": predicted_texts,
    "actual_number": test_data["ans"],
    "full_output": full_outputs
})

# Save results
results_df.to_csv("claim_mistral_2407_instruct.csv", index=False)
print("Predictions saved to 'claim_mistral_2407_instruct.csv'")

