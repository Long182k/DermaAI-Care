import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Separator } from "@/components/ui/separator";
import {
  Sheet,
  SheetContent,
  SheetHeader,
  SheetTitle,
  SheetTrigger,
} from "@/components/ui/sheet";
import { Camera, Image as ImageIcon, Send, Video, X } from "lucide-react";
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";
import { format } from "date-fns";

// Mock data for contacts (replace with actual API data later)
const contacts = [
  {
    id: 1,
    name: "Dr. Sarah Johnson",
    role: "Dermatologist",
    avatar: "https://images.unsplash.com/photo-1494790108377-be9c29b29330",
    online: true,
  },
  {
    id: 2,
    name: "Dr. Michael Chen",
    role: "Dermatologist",
    avatar: "https://images.unsplash.com/photo-1472099645785-5658abf4ff4e",
    online: false,
  },
];

// Mock data for messages (replace with actual API data later)
const initialMessages = [
  {
    id: 1,
    senderId: 1,
    content: "Hello! How can I help you today?",
    timestamp: new Date(2024, 2, 10, 9, 30),
    type: "text",
  },
  {
    id: 2,
    senderId: "user",
    content: "Hi Dr. Johnson, I have a question about my treatment.",
    timestamp: new Date(2024, 2, 10, 9, 31),
    type: "text",
  },
];

// Mock data for shared media (replace with actual API data later)
const sharedMedia = [
  {
    id: 1,
    type: "image",
    url: "https://images.unsplash.com/photo-1649972904349-6e44c42644a7",
    timestamp: new Date(2024, 2, 10, 9, 32),
  },
  {
    id: 2,
    type: "image",
    url: "https://images.unsplash.com/photo-1581091226825-a6a2a5aee158",
    timestamp: new Date(2024, 2, 10, 9, 33),
  },
];

export const ChatInterface = () => {
  const [messages, setMessages] = useState(initialMessages);
  const [newMessage, setNewMessage] = useState("");
  const [selectedContact, setSelectedContact] = useState(contacts[0]);
  const [isVideoCallActive, setIsVideoCallActive] = useState(false);

  const handleSendMessage = () => {
    if (!newMessage.trim()) return;

    const newMsg = {
      id: messages.length + 1,
      senderId: "user",
      content: newMessage,
      timestamp: new Date(),
      type: "text",
    };

    setMessages([...messages, newMsg]);
    setNewMessage("");
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  return (
    <div className="grid grid-cols-12 gap-4 h-[calc(100vh-12rem)]">
      {/* Contacts List */}
      <div className="col-span-3 bg-card rounded-lg p-4 border">
        <h2 className="font-semibold mb-4">Contacts</h2>
        <ScrollArea className="h-[calc(100vh-16rem)]">
          {contacts.map((contact) => (
            <div
              key={contact.id}
              className={`flex items-center gap-3 p-3 rounded-lg cursor-pointer hover:bg-accent ${
                selectedContact.id === contact.id ? "bg-accent" : ""
              }`}
              onClick={() => setSelectedContact(contact)}
            >
              <div className="relative">
                <Avatar>
                  <AvatarImage src={contact.avatar} alt={contact.name} />
                  <AvatarFallback>{contact.name[0]}</AvatarFallback>
                </Avatar>
                <span
                  className={`absolute bottom-0 right-0 w-3 h-3 border-2 border-background rounded-full ${
                    contact.online ? "bg-green-500" : "bg-gray-400"
                  }`}
                />
              </div>
              <div>
                <p className="font-medium">{contact.name}</p>
                <p className="text-sm text-muted-foreground">{contact.role}</p>
              </div>
            </div>
          ))}
        </ScrollArea>
      </div>

      {/* Chat Area */}
      <div className="col-span-6 bg-card rounded-lg border flex flex-col">
        {/* Chat Header */}
        <div className="p-4 border-b flex items-center justify-between">
          <div className="flex items-center gap-3">
            <Avatar>
              <AvatarImage
                src={selectedContact.avatar}
                alt={selectedContact.name}
              />
              <AvatarFallback>{selectedContact.name[0]}</AvatarFallback>
            </Avatar>
            <div>
              <h3 className="font-semibold">{selectedContact.name}</h3>
              <p className="text-sm text-muted-foreground">
                {selectedContact.online ? "Online" : "Offline"}
              </p>
            </div>
          </div>
          <div className="flex gap-2">
            <Button
              variant="outline"
              size="icon"
              onClick={() => setIsVideoCallActive(!isVideoCallActive)}
            >
              {isVideoCallActive ? <X /> : <Video />}
            </Button>
            <Button variant="outline" size="icon">
              <Camera />
            </Button>
          </div>
        </div>

        {/* Messages Area */}
        <ScrollArea className="flex-1 p-4">
          <div className="space-y-4">
            {messages.map((message) => (
              <div
                key={message.id}
                className={`flex ${
                  message.senderId === "user" ? "justify-end" : "justify-start"
                }`}
              >
                <div
                  className={`max-w-[70%] rounded-lg p-3 ${
                    message.senderId === "user"
                      ? "bg-primary text-primary-foreground"
                      : "bg-muted"
                  }`}
                >
                  <p>{message.content}</p>
                  <p className="text-xs mt-1 opacity-70">
                    {format(message.timestamp, "HH:mm")}
                  </p>
                </div>
              </div>
            ))}
          </div>
        </ScrollArea>

        {/* Input Area */}
        <div className="p-4 border-t">
          <div className="flex gap-2">
            <Input
              value={newMessage}
              onChange={(e) => setNewMessage(e.target.value)}
              onKeyPress={handleKeyPress}
              placeholder="Type your message..."
              className="flex-1"
            />
            <Button
              variant="secondary"
              size="icon"
              onClick={() => {
                // Handle image upload
              }}
            >
              <ImageIcon />
            </Button>
            <Button onClick={handleSendMessage}>
              <Send className="mr-2" />
              Send
            </Button>
          </div>
        </div>
      </div>

      {/* Media Sidebar */}
      <div className="col-span-3 bg-card rounded-lg p-4 border">
        <h2 className="font-semibold mb-4">Shared Media</h2>
        <ScrollArea className="h-[calc(100vh-16rem)]">
          <div className="grid grid-cols-2 gap-2">
            {sharedMedia.map((media) => (
              <div
                key={media.id}
                className="aspect-square rounded-lg overflow-hidden"
              >
                <img
                  src={media.url}
                  alt={`Shared media ${media.id}`}
                  className="w-full h-full object-cover"
                />
              </div>
            ))}
          </div>
        </ScrollArea>
      </div>

      {/* Video Call Sheet */}
      <Sheet open={isVideoCallActive} onOpenChange={setIsVideoCallActive}>
        <SheetContent side="right" className="w-[400px] sm:w-[540px]">
          <SheetHeader>
            <SheetTitle>Video Call with {selectedContact.name}</SheetTitle>
          </SheetHeader>
          <div className="mt-4">
            <div className="aspect-video bg-muted rounded-lg flex items-center justify-center">
              <p className="text-muted-foreground">Video call interface would be implemented here</p>
            </div>
            <div className="mt-4 flex justify-center gap-4">
              <Button variant="outline" size="icon">
                <Camera />
              </Button>
              <Button
                variant="destructive"
                size="icon"
                onClick={() => setIsVideoCallActive(false)}
              >
                <X />
              </Button>
            </div>
          </div>
        </SheetContent>
      </Sheet>
    </div>
  );
};