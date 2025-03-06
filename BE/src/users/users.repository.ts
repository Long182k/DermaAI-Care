import { Injectable, NotFoundException } from '@nestjs/common';
import { Prisma, User } from '@prisma/client';
import * as argon from 'argon2';
import { PrismaService } from '../prisma.service';
import { CreateUserDTO } from './dto/create-user.dto';
import { GetUserByKeywordDTO } from './dto/get-user.dto';
import { UpdateHashedRefreshTokenDTO } from './dto/update-user.dto';

@Injectable()
export class UserRepository {
  constructor(private prisma: PrismaService) {}

  async findUserByEmail(email: string): Promise<User | null> {
    {
      return await this.prisma.user.findUnique({
        where: { email },
      });
    }
  }

  async findUserByKeyword(keyword: GetUserByKeywordDTO): Promise<User | null> {
    const { userName, id, email, avatarUrl, bio } = keyword;
    {
      const user = await this.prisma.user.findFirst({
        where: {
          OR: [{ userName }, { id }, { email }, { avatarUrl }],
        },
      });

      if (!user) {
        throw new NotFoundException('User not found.');
      }

      return user;
    }
  }

  async findAllUsers(userId: string): Promise<User[]> {
    return this.prisma.user.findMany({
      where: {
        NOT: {
          id: userId,
        },
      },
    });
  }

  async createUser(data: CreateUserDTO): Promise<User> {
    data.avatarUrl =
      'https://res.cloudinary.com/dcivdqyyj/image/upload/v1736957755/sq1svii2veo8hewyelud.jpg';

    const {
      userName,
      password,
      email,
      avatarUrl,
      dateOfBirth,
      experience,
      education,
      certifications,
      languages,
      role,
    } = data;

    const hashedPassword = await argon.hash(password);

    const result = await this.prisma.user.create({
      data: {
        userName,
        email,
        password,
        hashedPassword,
        avatarUrl,
        dateOfBirth,
        experience,
        education,
        certifications,
        languages,
        role,
      },
    });

    delete result.hashedPassword;

    return result;
  }

  async updateHashedRefreshToken(
    params: UpdateHashedRefreshTokenDTO,
  ): Promise<User> {
    const { hashedRefreshToken, userId } = params;
    return this.prisma.user.update({
      where: { id: userId },
      data: {
        hashedRefreshToken,
      },
    });
  }

  async deleteUser(where: Prisma.UserWhereUniqueInput): Promise<User> {
    return this.prisma.user.delete({
      where,
    });
  }

  async update(userId: string, data: Partial<User>) {
    if (data.userName) {
      const existingUserName = await this.prisma.user.findUnique({
        where: { userName: data.userName },
      });

      if (existingUserName && existingUserName.id !== userId) {
        throw new NotFoundException('User name already exist in system.');
      }
    }

    const updateData: Partial<User> = {};

    if (data.userName !== undefined) updateData.userName = data.userName;
    if (data.dateOfBirth !== undefined)
      updateData.dateOfBirth = new Date(data.dateOfBirth);
    if (data.avatarUrl !== undefined) updateData.avatarUrl = data.avatarUrl;

    const result = await this.prisma.user.update({
      where: { id: userId },
      data: updateData,
    });

    return {
      ...result,
      userId,
    };
  }
}
